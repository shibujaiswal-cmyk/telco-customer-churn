# src/prepare_data.py
"""
Prepare data for Customer Churn project.
Usage:
    python src/prepare_data.py
Creates:
    data/processed.csv
"""
import os
import sys
import pandas as pd
import numpy as np

DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
OUTPUT_PATH = os.path.join("data", "processed.csv")

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}\nPlease put Telco-Customer-Churn.csv into the data/ folder.")
    # read with low_memory to avoid dtype warnings
    return pd.read_csv(path, low_memory=False)

def clean_data(df):
    # Drop customerID if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric (some rows are empty strings)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Standardize Churn column (target)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    else:
        raise KeyError("Expected column 'Churn' not found in dataset.")

    # Convert SeniorCitizen (if numeric) to string so get_dummies handles it consistently
    if 'SeniorCitizen' in df.columns and np.issubdtype(df['SeniorCitizen'].dtype, np.number):
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Strip whitespace from object columns (common dirty data)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Identify categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in num_cols:
        num_cols.remove('Churn')

    # One-hot encode categorical columns
    if len(cat_cols) > 0:
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df.copy()

    # Ensure columns are sorted (optional)
    df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)
    return df_encoded

def main():
    print("1) Loading data...")
    try:
        df = load_data(DATA_PATH)
        print(f"   Loaded dataset with shape: {df.shape}")
    except Exception as e:
        print("ERROR while loading dataset:")
        print(e)
        sys.exit(1)

    print("2) Cleaning and encoding data...")
    try:
        df_clean = clean_data(df)
        print(f"   After cleaning shape: {df_clean.shape}")
    except Exception as e:
        print("ERROR during cleaning:")
        print(e)
        sys.exit(1)

    print(f"3) Saving processed data -> {OUTPUT_PATH}")
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df_clean.to_csv(OUTPUT_PATH, index=False)
        print("   Saved processed.csv successfully.")
        print("Done. Next: run `python src/train_model.py` to train and save the model.")
    except Exception as e:
        print("ERROR while saving processed data:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
