# app/app.py
import streamlit as st
import joblib
import json
import numpy as np
import os
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor (30-feature dummy)", layout="centered")

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.json"

def load_artifacts():
    # clear, actionable errors if missing
    if not MODEL_PATH.exists():
        st.error(f"Model artifact not found at {MODEL_PATH}. Run src/create_dummy_30f_model.py or provide model artifacts.")
        st.stop()
    if not SCALER_PATH.exists():
        st.error(f"Scaler artifact not found at {SCALER_PATH}. Run src/create_dummy_30f_model.py first.")
        st.stop()
    if not FEATURES_PATH.exists():
        st.error(f"Features file not found at {FEATURES_PATH}. Run src/create_dummy_30f_model.py first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)["features"]
    return model, scaler, features

@st.cache_resource
def get_model():
    return load_artifacts()

model, scaler, feature_names = get_model()

st.title("Customer Churn Prediction — Demo (30 features)")
st.markdown("This demo uses a dummy model trained on synthetic data with 30 features. Use sliders / checkboxes and click **Predict Churn**.")

# Show feature count
st.write(f"Model expects **{len(feature_names)}** features.")

# Build inputs dynamically:
# For the first 10 numeric-style features, show numeric inputs (sliders)
# For the next 10 cat-style features, show selectboxes (simulate categories)
# For the last 10 flag-style features, show checkboxes (0/1)
user_inputs = {}

# Numeric features
for fname in feature_names[:10]:
    # reasonable ranges
    user_inputs[fname] = st.number_input(fname, min_value=-10.0, max_value=100.0, value=0.0, step=0.1)

# Cat-like features (simulate 3 categories each)
for fname in feature_names[10:20]:
    # derive a simple set of options
    opts = ["A", "B", "C"]
    # default "A"
    sel = st.selectbox(fname, opts, index=0)
    # convert category to one-hot-like encoded set of three virtual columns
    # but to keep feature vector length consistent with features.json,
    # we will map A->0, B->1, C->2 as numeric proxies (dummy model trained on numeric inputs)
    # (This matches created dummy features which are numeric)
    user_inputs[fname] = 0 if sel == "A" else (1 if sel == "B" else 2)

# Flag features: checkboxes (0/1)
for fname in feature_names[20:30]:
    user_inputs[fname] = 1 if st.checkbox(fname, value=False) else 0

# Build feature vector in the exact order of feature_names
def build_feature_vector(feature_names, user_inputs):
    vec = []
    for f in feature_names:
        # fallback to 0 if user didn't provide
        vec.append(user_inputs.get(f, 0))
    return np.array(vec).reshape(1, -1).astype(float)

if st.button("Predict Churn"):
    try:
        X = build_feature_vector(feature_names, user_inputs)
        # scale and predict
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction")
        if proba is not None:
            st.write(f"🔹 Churn Probability: **{proba*100:.2f}%**")
        if pred == 1:
            st.error("⚠️ This customer is likely to **churn**.")
        else:
            st.success("✅ This customer is likely to **stay**.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        # Provide additional debug info (non-sensitive)
        st.write("Feature vector shape:", X.shape)
        st.write("Expected scaler input dim:", getattr(scaler, "n_features_in_", "unknown"))
