# src/train_model.py
"""
Robust Train script for Customer Churn project.

Handles cases where processed CSV still contains string/categorical columns by
applying one-hot encoding before training.

Usage:
    python src/train_model.py
"""
import os
import sys
import glob
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_DIR = "data"
MODEL_DIR = "model"
RANDOM_STATE = 42

def find_csv_file(data_dir):
    preferred = os.path.join(data_dir, "processed.csv")
    if os.path.exists(preferred):
        return preferred
    csvs = glob.glob(os.path.join(data_dir, "*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'. Please run prepare_data.py or place the processed CSV there.")
    for c in csvs:
        name = os.path.basename(c).lower()
        if "processed" in name or "clean" in name:
            return c
    return csvs[0]

def load_data(path):
    return pd.read_csv(path, low_memory=False)

def ensure_numeric_features(df, target_col="Churn"):
    """
    Ensures all feature columns are numeric:
    - Leaves target column alone
    - Converts object/category columns by pd.get_dummies (drop_first=True)
    - Fills any remaining NaNs with 0
    Returns: X_df (numeric), y_series
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # detect object/categorical cols
    obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        print(f"Found categorical columns to encode: {obj_cols}")
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # After get_dummies, ensure all columns are numeric
    # If some columns still aren't numeric (rare), try coercion
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        print("Warning: Some columns still non-numeric; attempting coercion:", non_numeric)
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # Fill any residual NaNs
    if X.isnull().sum().sum() > 0:
        print("Filling NaN values in features with 0")
        X = X.fillna(0)

    return X, y

def main():
    print("1) Locating processed CSV in data/ ...")
    try:
        csv_path = find_csv_file(DATA_DIR)
        print(f"   Using file: {csv_path}")
    except Exception as e:
        print("ERROR finding CSV:", e)
        sys.exit(1)

    try:
        df = load_data(csv_path)
        print(f"   Loaded data shape: {df.shape}")
    except Exception as e:
        print("ERROR loading CSV:", e)
        sys.exit(1)

    if 'Churn' not in df.columns:
        print("ERROR: 'Churn' column not found in processed data. Column names:", df.columns.tolist())
        sys.exit(1)

    print("2) Converting categorical features to numeric (if any)...")
    X, y = ensure_numeric_features(df, target_col='Churn')
    print(f"   Feature matrix shape after encoding: {X.shape}")

    # Final safety check
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs still present in X; filling with 0")
        X = X.fillna(0)

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("3) Training RandomForestClassifier ...")
    model = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("4) Evaluating model ...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_test, y_proba)
            print(f"   ROC-AUC: {roc:.4f}")
        except Exception:
            print("   ROC-AUC: could not compute.")
    print("   Classification report:")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "features.json")

    print(f"5) Saving model -> {model_path}")
    joblib.dump(model, model_path)
    print(f"   Saving scaler -> {scaler_path}")
    joblib.dump(scaler, scaler_path)

    features = X.columns.tolist()
    with open(features_path, "w") as f:
        json.dump({"features": features}, f)
    print(f"   Saved features list -> {features_path}")
    print("Done. Artifacts saved in model/. Now run: streamlit run app/app.py")

if __name__ == "__main__":
    main()
