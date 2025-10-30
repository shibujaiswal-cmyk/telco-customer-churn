# app/app.py
"""
Customer Churn Prediction — Streamlit app (clean, defensive).
Assumes the following files exist in `model/`:
  - model.pkl
  - scaler.pkl
  - features.json   (json: {"features": [<ordered feature names>]})

This file is safe to commit to GitHub now. You can deploy later when ready.
"""
import streamlit as st
import joblib
import json
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Paths for artifacts (relative to repo root / app working dir)
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"

def load_artifacts():
    """Load model, scaler and feature order. Provide clear error messages if missing."""
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not SCALER_PATH.exists():
        missing.append(str(SCALER_PATH))
    if not FEATURES_PATH.exists():
        missing.append(str(FEATURES_PATH))

    if missing:
        st.error("Missing model artifacts. Please ensure the following files exist in the repository's `model/` folder:")
        for m in missing:
            st.write(f"- {m}")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model at {MODEL_PATH}: {e}")
        st.stop()

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to load scaler at {SCALER_PATH}: {e}")
        st.stop()

    try:
        with open(FEATURES_PATH, "r") as f:
            features = json.load(f).get("features", [])
    except Exception as e:
        st.error(f"Failed to load features.json at {FEATURES_PATH}: {e}")
        st.stop()

    if not isinstance(features, list) or len(features) == 0:
        st.error("features.json must contain a non-empty list under key 'features'.")
        st.stop()

    return model, scaler, features

@st.cache_resource
def get_model():
    return load_artifacts()

# Load artifacts (or stop with an instructive error)
model, scaler, feature_names = get_model()

st.title("Customer Churn Prediction")
st.markdown("Enter customer data below and click **Predict Churn**. (This app expects the model artifacts in `model/`.)")

st.write(f"Model expects **{len(feature_names)}** features.")

# Build a simple dynamic input UI based on feature name patterns.
# We present sensible default input widgets:
# - numeric-like: number_input
# - features containing 'flag' or 'yes' -> checkbox
# - features containing 'cat' or 'payment' -> selectbox with simple options
# You can customize these heuristics later.

user_inputs = {}

def is_numeric_name(name):
    # heuristics: 'num' or 'feature_' or 'charge' or 'total' or 'monthly' or 'tenure'
    lower = name.lower()
    return any(tok in lower for tok in ["num", "feature_", "charge", "monthly", "total", "tenure", "age"])

def is_flag_name(name):
    lower = name.lower()
    return any(tok in lower for tok in ["flag", "yes", "no", "true", "false", "has_"])

def is_cat_name(name):
    lower = name.lower()
    return any(tok in lower for tok in ["cat", "contract", "payment", "internet", "service", "method"])

# Create inputs in three columns for better layout
cols = st.columns(3)
for i, fname in enumerate(feature_names):
    col = cols[i % 3]
    with col:
        label = fname.replace("_", " ").title()
        if is_numeric_name(fname):
            # numeric input: reasonable default range
            user_inputs[fname] = st.number_input(label, value=0.0, step=0.1, format="%.3f")
        elif is_flag_name(fname):
            user_inputs[fname] = 1 if st.checkbox(label, value=False) else 0
        elif is_cat_name(fname):
            # simple category choices; map them to numeric proxies
            opt = st.selectbox(label, options=["A", "B", "C"], index=0)
            user_inputs[fname] = 0 if opt == "A" else (1 if opt == "B" else 2)
        else:
            # default numeric fallback
            user_inputs[fname] = st.number_input(label, value=0.0, step=0.1, format="%.3f")

# Build feature vector in the same order as features.json
def build_feature_vector(feature_names, inputs):
    vec = []
    for f in feature_names:
        vec.append(inputs.get(f, 0))
    return np.array(vec).reshape(1, -1).astype(float)

# Predict button
if st.button("Predict Churn"):
    X = build_feature_vector(feature_names, user_inputs)
    # Validate scaler/model input dimension
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and expected != X.shape[1]:
        st.error(f"Feature dimension mismatch: the loaded scaler expects {expected} features but the input has {X.shape[1]}.")
        st.write("If you used a dummy model, regenerate model/scaler/features.json so they match this app's feature list.")
    else:
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

            st.subheader("Prediction Result")
            if proba is not None:
                st.write(f"Churn Probability: **{proba*100:.2f}%**")
            if pred == 1:
                st.error("⚠️ Likely to churn")
            else:
                st.success("✅ Likely to stay")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Feature vector shape:", X.shape)
            st.write("Scaler expects:", expected)
