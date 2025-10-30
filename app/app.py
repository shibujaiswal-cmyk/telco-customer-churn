# app/app.py
# --- AUTO DOWNLOAD MODEL FROM GITHUB RELEASE ---
import os
import requests
from pathlib import Path

# 🔗 Replace this URL with your own GitHub Release asset link
MODEL_RELEASE_URL = "https://github.com/shibujaiswal-cmyk/telco-customer-churn/releases/download/v1.0/model.pkl"

# Make sure 'model' folder exists
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)

# Local model path
local_model_path = model_dir / "model.pkl"

# Download model if missing
if not local_model_path.exists():
    try:
        print("Downloading model from GitHub Release...")
        with requests.get(MODEL_RELEASE_URL, stream=True) as r:
            r.raise_for_status()
            with open(local_model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Model downloaded successfully.")
    except Exception as e:
        print("⚠️ Model download failed:", e)
# --- END AUTO DOWNLOAD ---

# At top of app/app.py (add these imports)
import os
import requests
from pathlib import Path

# CONFIG: paste the Release asset URL you copied
MODEL_RELEASE_URL = None

# Ensure model folder exists and model is present (download if missing)
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)
local_model_path = model_dir / "model.pkl"

if not local_model_path.exists():
    try:
        # Stream download to avoid memory spike
        with requests.get(MODEL_RELEASE_URL, stream=True) as r:
            r.raise_for_status()
            with open(local_model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Downloaded model to", local_model_path)
    except Exception as e:
        print("Failed to download model:", e)

import streamlit as st
import joblib
import json
import numpy as np
import os
import pandas as pd

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

MODEL_DIR = os.path.join("..", "model") if os.path.exists(os.path.join("..","model")) else "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.json")

def load_artifacts():
    # load model, scaler, features; show clear errors if missing
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Run training script first.")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler not found at {SCALER_PATH}. Run training script first.")
        st.stop()
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Features file not found at {FEATURES_PATH}. Run training script first.")
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

st.title("Customer Churn Prediction")
st.markdown("Enter customer details below. App will create a feature vector in the trained feature order.")

# Friendly info about available features
st.write(f"Model expects **{len(feature_names)}** features. Showing a compact input form for fastest use.")

# Helper: find common numeric features if present
def find_feature(name_substrs):
    for s in name_substrs:
        for f in feature_names:
            if s.lower() in f.lower():
                return f
    return None

# Identify common numeric fields
tenure_feat = find_feature(["tenure"])
monthly_feat = find_feature(["monthlycharges", "monthly_charge", "monthly"])
total_feat = find_feature(["totalcharges", "total_charge", "total"])

# Input collection dictionary
user_inputs = {}

# Numeric inputs
if tenure_feat:
    user_inputs[tenure_feat] = st.number_input("Tenure (months)", min_value=0, max_value=200, value=12)
if monthly_feat:
    user_inputs[monthly_feat] = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0)
if total_feat:
    user_inputs[total_feat] = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=1000.0)

# For compactness, allow toggles for a few common one-hot features if they exist
# E.g., Contract_Two year, Contract_One year, PaperlessBilling_Yes, InternetService_Fiber optic, PaymentMethod_Bank transfer...
one_hot_candidates = [
    "Contract_Two year", "Contract_One year", "Contract_Month-to-month",
    "PaperlessBilling_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    "MultipleLines_No phone service"
]

# Lowercase mapping for quick checking
available_onehots = [f for f in feature_names if "_" in f and any(part.strip().lower() in f.lower() for part in ["contract", "paperless", "internetservice", "paymentmethod", "multiplelines"])]

# Show checkboxes for up to 6 available one-hot features
count = 0
for f in feature_names:
    if count >= 6:
        break
    # simple heuristic: include feature if it looks like a one-hot column (contains '_' or common keywords)
    if any(k in f.lower() for k in ["contract", "paperless", "internetservice", "paymentmethod", "multiplelines", "gender", "seniorcitizen", "partner", "dependents"]):
        # display a nicer label
        label = f.replace("_", " ")
        user_inputs[f] = st.checkbox(label, value=False)
        count += 1

# If no inputs were auto-detected, provide a generic numeric input for the first 3 features
if len(user_inputs) == 0:
    st.warning("Could not detect common features automatically. Please provide values for first 3 features manually.")
    for i, feat in enumerate(feature_names[:3]):
        user_inputs[feat] = st.number_input(feat, value=0.0)

# Assemble feature vector in correct order
def build_feature_vector(feature_names, user_inputs):
    vec = []
    for f in feature_names:
        if f in user_inputs:
            vec.append(user_inputs[f])
        else:
            # default for unspecified features = 0
            vec.append(0)
    return np.array(vec).reshape(1, -1)

# Button for prediction
if st.button("Predict Churn"):
    try:
        X = build_feature_vector(feature_names, user_inputs)
        # ensure numeric type
        X = X.astype(float)
        # scale
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
        st.stop()

# Footer - show model metrics if present (optional)
with st.expander("Show model feature sample (first 10 features)"):
    st.write(feature_names[:10])

