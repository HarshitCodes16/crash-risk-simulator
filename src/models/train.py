"""
Two-Stage Model Training
-------------------------
Stage 1: Crash Classifier   -> predicts probability of a crash
Stage 2: Severity Regressor -> predicts impact force, trained only on crash rows

Both stages compare multiple algorithms and keep the best one (by ROC-AUC for
classification, R2 for regression), logging a comparison table + metrics so
the choice is justified rather than arbitrary.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score
)

FEATURES = ["speed", "distance", "reaction_time", "brake_eff", "friction", "mass"]
SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "crash_simulation_data_v2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def train_stage1_classifier(df):
    X = df[FEATURES]
    y = df["crash"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }

    comparison = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        comparison.append(row)
        fitted[name] = model

    comparison_df = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted[best_name]

    # Confusion matrix for the best model
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds)
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Crash", "Crash"]).plot(ax=ax, cmap="Blues")
        plt.title(f"Confusion Matrix - {best_name} (Stage 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "stage1_confusion_matrix.png"))
        plt.close()
    except OSError:
        pass  # read-only filesystem (e.g. some cloud environments) - safe to skip

    return best_model, best_name, comparison_df


def train_stage2_regressor(df):
    crash_df = df[df["crash"] == 1]
    X = crash_df[FEATURES]
    y = crash_df["impact_force"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    }

    comparison = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        row = {
            "model": name,
            "r2": r2_score(y_test, preds),
            "rmse": mean_squared_error(y_test, preds) ** 0.5,
            "mae": mean_absolute_error(y_test, preds),
        }
        comparison.append(row)
        fitted[name] = model

    comparison_df = pd.DataFrame(comparison).sort_values("r2", ascending=False)
    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted[best_name]

    return best_model, best_name, comparison_df


def plot_feature_importance(model, model_name, stage_label):
    if not hasattr(model, "feature_importances_"):
        return
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([FEATURES[i] for i in order], importances[order], color="#D85A30")
        ax.set_title(f"Feature Importance - {model_name} ({stage_label})")
        plt.xticks(rotation=30)
        plt.tight_layout()
        fname = f"{stage_label.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(os.path.join(RESULTS_DIR, fname))
        plt.close()
    except OSError:
        pass  # read-only filesystem (e.g. some cloud environments) - safe to skip


def train_fast_fallback(df):
    """
    Lightweight training path used ONLY as a runtime fallback (e.g. when a
    deployment environment's scikit-learn version can't unpickle the models
    trained locally). Skips the full 4-vs-4 model comparison and trains just
    the single best-known model per stage directly, on a smaller sample, so
    cold-start on constrained cloud CPUs stays fast (a few seconds instead of
    minutes).
    """
    X = df[FEATURES]
    y_clf = df["crash"]

    clf = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=SEED)
    clf.fit(X, y_clf)

    crash_df = df[df["crash"] == 1]
    reg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=SEED)
    reg.fit(crash_df[FEATURES], crash_df["impact_force"])

    return clf, reg


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Crash rate: {df['crash'].mean():.2%}")

    print("\n--- Stage 1: Crash Classifier ---")
    clf, clf_name, clf_comparison = train_stage1_classifier(df)
    print(clf_comparison.to_string(index=False))
    print(f"Best Stage 1 model: {clf_name}")
    plot_feature_importance(clf, clf_name, "Stage 1 Classifier")

    print("\n--- Stage 2: Severity Regressor ---")
    reg, reg_name, reg_comparison = train_stage2_regressor(df)
    print(reg_comparison.to_string(index=False))
    print(f"Best Stage 2 model: {reg_name}")
    plot_feature_importance(reg, reg_name, "Stage 2 Regressor")

    joblib.dump(clf, os.path.join(MODELS_DIR, "stage1_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODELS_DIR, "stage2_regressor.pkl"))

    metrics_summary = {
        "stage1_best_model": clf_name,
        "stage1_comparison": clf_comparison.to_dict(orient="records"),
        "stage2_best_model": reg_name,
        "stage2_comparison": reg_comparison.to_dict(orient="records"),
        "features": FEATURES,
        "seed": SEED,
    }
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}")
    print(f"Saved metrics + plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
