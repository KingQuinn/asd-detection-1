import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "reports")


def show_explore_page():
    st.title("ASD Detection System")
    st.write(
        "This app uses a stacking ensemble (SVM, k-NN and Decision Tree base models "
        "with an XGBoost meta-model) to screen for Autism Spectrum Disorder (ASD) traits."
    )

    metrics_path = os.path.join(REPORT_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        st.info("Run `python train.py` to generate model metrics and figures.")
        return

    with open(metrics_path) as f:
        report = json.load(f)
    st.subheader("Model performance (held-out test set)")
    table = pd.DataFrame(report["test_metrics"]).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    st.dataframe(table.astype(float).style.format("{:.3f}"))

    figures = os.path.join(REPORT_DIR, "figures")
    for name, caption in [
        ("model_performance_comparison.png", "Model comparison"),
        ("roc_curves.png", "ROC curves"),
        ("confusion_matrix_xgboost_stacking.png", "Confusion matrix: stacked model"),
        ("feature_importance.png", "Feature importance"),
        ("correlation_matrix.png", "Correlation matrix"),
    ]:
        path = os.path.join(figures, name)
        if os.path.exists(path):
            st.image(path, caption=caption)
