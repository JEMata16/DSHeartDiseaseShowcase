"""
Heart disease risk modelling: exploratory analysis, model comparison and evaluation.

Run from the showcase folder:
    python analysis/heart_analysis.py

Outputs:
    figures/   - EDA and evaluation plots (PNG)
    results/   - metrics.json and model_comparison.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", message="The `probability` parameter")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal", 3: "Asymptomatic"},
}

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_before} rows, dropped {n_before - len(df)} duplicate(s) -> {len(df)} rows")
    return df


def eda(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "features": len(FEATURES),
        "positive_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
    }

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["No disease", "Disease"], counts.values, color=[PALETTE[0], PALETTE[1]])
    ax.bar_label(bars, padding=3)
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Numeric distributions split by outcome
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(numeric), figsize=(18, 4))
    for ax, col in zip(axes, numeric):
        for t in (0, 1):
            sns.kdeplot(df.loc[df[TARGET] == t, col], ax=ax, fill=True, alpha=0.4,
                        color=PALETTE[t], label=["No disease", "Disease"][t])
        ax.set_title(col)
        ax.set_xlabel("")
    axes[0].legend()
    fig.suptitle("Clinical measures by outcome", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Disease rate by categorical features
    cats = ["cp", "sex", "exang", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cats):
        rate = df.groupby(col)[TARGET].agg(["mean", "size"]).reset_index()
        rate = rate[rate["size"] >= 5]  # hide tiny groups that make rates unstable
        ax.bar(rate[col].astype(str), rate["mean"], color="#55A868")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {col}")
        ax.set_ylabel("Share with disease")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_disease_rate_by_category.png", dpi=150)
    plt.close(fig)

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    summary["top_target_correlations"] = (
        corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(5).round(3).to_dict()
    )
    return summary


def build_candidates() -> dict:
    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    return {
        "Logistic Regression": (
            scaled(LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            scaled(SVC(probability=True, random_state=SEED)),
            {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.05]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED),
            {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"max_depth": [1, 2, 3], "learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
        ),
    }


def compare_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}

    baseline = DummyClassifier(strategy="most_frequent")
    base_scores = _cv_scores(baseline, X_train, y_train, cv)
    rows.append({"model": "Majority-class baseline", **base_scores, "params": "-"})

    for name, (estimator, grid) in build_candidates().items():
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        scores = _cv_scores(best, X_train, y_train, cv)
        rows.append({"model": name, **scores, "params": json.dumps(search.best_params_)})
        fitted[name] = best

    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    table.to_csv(RES_DIR / "model_comparison.csv", index=False)
    print(table.round(3).to_string(index=False))
    return table, fitted


def _cv_scores(model, X, y, cv) -> dict:
    out = {}
    for metric in ["roc_auc", "accuracy", "recall", "f1"]:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        out[f"cv_{metric}"] = round(float(scores.mean()), 4)
        out[f"cv_{metric}_std"] = round(float(scores.std()), 4)
    return out


def evaluate_best(name, model, X_train, X_test, y_train, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": name,
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred), 4),
        "test_f1": round(f1_score(y_test, pred), 4),
    }

    # Bootstrap 95% CI for ROC-AUC, so the headline number has an honest range
    rng = np.random.default_rng(SEED)
    boots = []
    y_arr = np.asarray(y_test)
    for _ in range(1000):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_arr[idx], proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    metrics["test_roc_auc_95ci"] = [round(float(lo), 4), round(float(hi), 4)]

    # Figure: ROC curve + confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=axes[0], curve_kwargs={"color": PALETTE[1]})
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curve (held-out test set)")
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                           display_labels=["No disease", "Disease"]).plot(
        ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Confusion matrix at 0.5 threshold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_best_model_evaluation.png", dpi=150)
    plt.close(fig)

    # Figure: permutation importance on the test set
    imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=SEED,
                                 scoring="roc_auc")
    importance = (pd.Series(imp.importances_mean, index=X_test.columns)
                  .sort_values(ascending=True))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance.index, importance.values, color="#8172B2")
    ax.set_title(f"Permutation importance (drop in ROC-AUC), {name}")
    ax.set_xlabel("Mean decrease in ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_feature_importance.png", dpi=150)
    plt.close(fig)
    metrics["top_features"] = importance.sort_values(ascending=False).head(5).round(4).to_dict()

    return metrics


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    eda_summary = eda(df)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    table, fitted = compare_models(X_train, y_train)

    best_name = table.iloc[0]["model"]
    if best_name == "Majority-class baseline":
        raise SystemExit("Baseline ranked first; check the candidate models.")
    best_model = fitted[best_name]
    best_model.fit(X_train, y_train)

    final = evaluate_best(best_name, best_model, X_train, X_test, y_train, y_test)
    report = {
        "dataset": eda_summary,
        "split": {"train": len(X_train), "test": len(X_test), "stratified": True, "seed": SEED},
        "selection_metric": "5-fold stratified CV ROC-AUC on the training split",
        "best_model": final,
        "comparison": table.to_dict(orient="records"),
    }
    (RES_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
