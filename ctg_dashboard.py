#!/usr/bin/env python3
"""
CTG ML Pipeline – Results & Prediction Dashboard
==================================================
Streamlit app to browse results produced by ctg_pipeline.py and run
live predictions with a saved model bundle. Run with:

    streamlit run ctg_dashboard.py
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CTG ML Pipeline Dashboard", layout="wide")

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
DATA_PATH = "CTG_dataset.csv"

CLASS_LABELS = ["Normal", "Suspect", "Pathologic"]


@st.cache_data
def load_dataset_stats(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    X = df.drop(columns=["NSP"])
    stats = pd.DataFrame({
        "min": X.min(),
        "max": X.max(),
        "mean": X.mean(),
    })
    return stats


@st.cache_data
def list_files(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    return sorted(files, key=os.path.getmtime, reverse=True)


@st.cache_resource
def load_model_bundle(path):
    return joblib.load(path)


def predict_single(bundle, input_values):
    X = pd.DataFrame([input_values])[bundle["feature_names"]]
    X_scaled = bundle["scaler"].transform(X)
    if bundle.get("selector") is not None:
        X_scaled = bundle["selector"].transform(X_scaled)
    model = bundle["model"]
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    return pred_idx, proba


st.title("CTG ML Pipeline Dashboard")

page = st.sidebar.radio("View", ["Results Overview", "Live Prediction"])

if page == "Results Overview":
    st.header("Model comparison")

    metrics_files = list_files(RESULTS_DIR, "metrics_*.csv")
    if not metrics_files:
        st.warning("No metrics CSV found in results/. Run ctg_pipeline.py first.")
    else:
        labels = [os.path.basename(f) for f in metrics_files]
        choice = st.selectbox("Run", labels)
        df = pd.read_csv(metrics_files[labels.index(choice)])

        numeric_cols = df.select_dtypes(include="number").columns
        best_row = df["Accuracy"].idxmax()
        st.dataframe(
            df.style.format({c: "{:.4f}" for c in numeric_cols})
              .highlight_max(subset=["Accuracy", "F1", "AUC"], color="#2e7d3222"),
            width="stretch",
        )
        st.caption(f"Best by accuracy: **{df.loc[best_row, 'Model']}** "
                   f"({df.loc[best_row, 'Accuracy']:.2%})")

    st.header("Plots")
    st.caption("Plots reflect the most recent pipeline run (overwritten each run).")
    col1, col2 = st.columns(2)
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    if os.path.exists(cm_path):
        col1.image(cm_path, caption="Confusion Matrix", width="stretch")
    else:
        col1.info("No confusion matrix yet.")
    if os.path.exists(fi_path):
        col2.image(fi_path, caption="Feature Importance", width="stretch")
    else:
        col2.info("No feature importance plot yet.")

else:
    st.header("Live prediction")

    model_files = list_files(MODELS_DIR, "*.pkl")
    if not model_files:
        st.warning("No saved model found in models/. Run ctg_pipeline.py first "
                    "(it saves the best model automatically).")
    else:
        labels = [os.path.basename(f) for f in model_files]
        choice = st.selectbox("Model", labels)
        bundle = load_model_bundle(model_files[labels.index(choice)])
        st.caption(f"Loaded: **{bundle['model_name']}** "
                   f"({'selected features: ' + str(bundle['selected_features']) if bundle.get('selected_features') else 'all features'})")

        stats = load_dataset_stats(DATA_PATH)

        st.subheader("Input features")
        st.caption("Sliders default to the dataset mean; range is the dataset min/max.")
        input_values = {}
        feature_names = bundle["feature_names"]
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            row = stats.loc[feat]
            with cols[i % 3]:
                input_values[feat] = st.slider(
                    feat,
                    min_value=float(row["min"]),
                    max_value=float(row["max"]),
                    value=float(row["mean"]),
                )

        if st.button("Predict", type="primary"):
            pred_idx, proba = predict_single(bundle, input_values)
            st.success(f"Predicted class: **{CLASS_LABELS[pred_idx]}**")
            proba_df = pd.DataFrame({
                "Class": bundle.get("class_labels", CLASS_LABELS),
                "Probability": proba,
            }).set_index("Class")
            st.bar_chart(proba_df)
