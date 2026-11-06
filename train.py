"""Train, evaluate and export the stacked ASD detection model.

Pipeline (matches the system architecture in the project report):
  1. Preprocess Autism_Dataset_Combined.csv (type conversion, categorical
     encoding, duplicate removal, Age outlier removal).
  2. Stratified 80/20 train/test split.
  3. First stack: tune k-NN, SVM and Decision Tree with GridSearchCV.
  4. New training set: 5-fold out-of-fold class probabilities of the base models.
  5. Second stack: tune an XGBoost meta-model on those probabilities.
  6. Evaluate on the held-out test set, save figures, metrics and the model.

Usage:  python train.py
"""
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "asd-backend")
sys.path.insert(0, BACKEND)
from asd_core import ENCODINGS, FEATURES  # noqa: E402

DATA_PATH = os.path.join(ROOT, "Autism_Dataset_Combined.csv")
CLEANED_PATH = os.path.join(ROOT, "cleaned_dataset.csv")
MODEL_PATHS = [os.path.join(BACKEND, "asd_detection2.pkl")]
REPORT_DIR = os.path.join(ROOT, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
SEED = 42


def preprocess(df):
    df = df.copy()
    for col in ["Sex", "Jauundice", "Family_ASD", "Class"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["Class"] = df["Class"].map({"no": 0, "yes": 1})
    for col, mapping in ENCODINGS.items():
        df[col] = df[col].map(mapping)
    if df.isna().any().any():
        raise ValueError("Unexpected categorical values in dataset")

    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    # Z-score outlier removal on Age (the only continuous feature).
    z = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    df = df[z.abs() < 3]
    print(f"Rows: {n0} raw, {n_dup} duplicates removed, {n0 - n_dup - len(df)} Age outliers removed, {len(df)} kept")
    return df.reset_index(drop=True)


def scaled(estimator):
    """Scale Age only (the AQ items and binary flags are already 0/1)."""
    pre = ColumnTransformer([("age", StandardScaler(), ["Age"])], remainder="passthrough")
    return Pipeline([("scale", pre), ("clf", estimator)])


def metrics_for(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = preprocess(pd.read_csv(DATA_PATH))
    df.to_csv(CLEANED_PATH, index=False)

    # Bivariate analysis
    plt.figure(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, annot_kws={"size": 7})
    plt.title("Correlation matrix of ASD features")
    save_fig("correlation_matrix.png")

    X, y = df[FEATURES], df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {X_train.shape}, test {X_test.shape}")

    rf = RandomForestClassifier(random_state=SEED).fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(FEATURES)[order], rf.feature_importances_[order], color="teal")
    plt.title("Feature importance (Random Forest, training set)")
    save_fig("feature_importance.png")

    # First stack: tune base learners
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    searches = {
        "SVM": GridSearchCV(
            scaled(SVC(probability=True, random_state=SEED)),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}, cv=cv, n_jobs=-1),
        "k-NN": GridSearchCV(
            scaled(KNeighborsClassifier()), {"clf__n_neighbors": [3, 5, 7, 9]}, cv=cv, n_jobs=-1),
        "Decision Tree": GridSearchCV(
            DecisionTreeClassifier(random_state=SEED),
            {"max_depth": [3, 5, 10], "min_samples_split": [2, 5, 10]}, cv=cv, n_jobs=-1),
    }
    base = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        base[name] = search.best_estimator_
        print(f"{name}: best {search.best_params_} (CV acc {search.best_score_:.4f})")

    # New training set: out-of-fold probabilities, so the meta-model never sees
    # predictions a base model made on its own training rows.
    oof = pd.DataFrame({
        name: cross_val_predict(est, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        for name, est in base.items()
    })

    # Second stack: tune the XGBoost meta-model on the stacked features
    xgb_search = GridSearchCV(
        XGBClassifier(random_state=SEED, eval_metric="logloss"),
        {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5],
         "gamma": [0, 0.1], "subsample": [0.7, 1.0]},
        cv=cv, n_jobs=-1)
    xgb_search.fit(oof, y_train)
    print(f"XGBoost meta-model: best {xgb_search.best_params_} (CV acc {xgb_search.best_score_:.4f})")

    stack = StackingClassifier(
        estimators=[(name.replace(" ", "_").replace("-", ""), est) for name, est in base.items()],
        final_estimator=XGBClassifier(random_state=SEED, eval_metric="logloss", **xgb_search.best_params_),
        cv=cv, stack_method="predict_proba", n_jobs=-1)
    stack.fit(X_train, y_train)

    # Evaluation on the held-out test set
    models = {**base, "XGBoost Stacking": stack}
    results, probas = {}, {}
    for name, model in models.items():
        pred = model.predict(X_test)
        probas[name] = model.predict_proba(X_test)[:, 1]
        results[name] = metrics_for(y_test, pred, probas[name])
        results[name]["train_accuracy"] = accuracy_score(y_train, model.predict(X_train))
        print(f"\n===== {name} =====\n{classification_report(y_test, pred, digits=3)}")

        plt.figure(figsize=(5, 4.5))
        sns.heatmap(results[name]["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Non-ASD", "ASD"], yticklabels=["Non-ASD", "ASD"])
        plt.title(f"Confusion matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        save_fig(f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '')}.png")

    plt.figure(figsize=(7, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves (test set)")
    plt.legend()
    save_fig("roc_curves.png")

    plt.figure(figsize=(7, 6))
    for name, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        plt.plot(rec, prec, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves (test set)")
    plt.legend()
    save_fig("precision_recall_curves.png")

    summary = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].astype(float)
    summary.plot(kind="bar", figsize=(9, 5), ylim=(0.8, 1.0), rot=0)
    plt.title("Model performance comparison (test set)")
    plt.legend(loc="lower right")
    save_fig("model_performance_comparison.png")
    print("\n", summary.round(4).to_string())

    report = {
        "dataset": {"rows_after_cleaning": len(df), "train": len(X_train), "test": len(X_test),
                    "asd_rate": float(y.mean())},
        "best_params": {**{n: s.best_params_ for n, s in searches.items()}, "XGBoost meta": xgb_search.best_params_},
        "test_metrics": results,
    }
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    bundle = {
        "model": stack,
        "features": FEATURES,
        "encodings": ENCODINGS,
        "metrics": results["XGBoost Stacking"],
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    for path in MODEL_PATHS:
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
