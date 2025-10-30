# src/create_dummy_30f_model.py
"""
Create a dummy RandomForest model trained on synthetic data with 30 features.
This writes:
 - model/model.pkl
 - model/scaler.pkl
 - model/features.json
Use this to make app/app.py and artifacts compatible (no shape mismatch).
"""
import os
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create model directory
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)

# PARAMETERS
N_FEATURES = 30
N_SAMPLES = 2000
RANDOM_STATE = 42

# Synthetic dataset
rng = np.random.default_rng(RANDOM_STATE)
X = rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, N_FEATURES))

# Create a binary target with some dependency on a few features
# e.g., target = 1 when sum of selected signal features + noise > threshold
signal_weights = np.zeros(N_FEATURES)
signal_indices = [0, 3, 5, 7, 11]  # some indices to create signal
for i in signal_indices:
    signal_weights[i] = rng.uniform(0.8, 1.5)
logits = X.dot(signal_weights) + rng.normal(0, 0.5, size=N_SAMPLES)
y = (logits > np.median(logits)).astype(int)

# Fit scaler and model
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(Xs, y)

# Save artifacts
joblib.dump(model, model_dir / "model.pkl")
joblib.dump(scaler, model_dir / "scaler.pkl")

# Create feature names (consistent order) — use sensible names
features = []
for i in range(N_FEATURES):
    # mix numeric-like names and some one-hot style names for UI variety
    if i < 10:
        features.append(f"num_feature_{i+1}")
    elif i < 20:
        features.append(f"cat_feature_{i-9}_A")  # example one-hot style token
    else:
        features.append(f"flag_feature_{i-19}_Yes")

with open(model_dir / "features.json", "w") as f:
    json.dump({"features": features}, f)

print("✅ Dummy 30-feature model created.")
print("Model path:", (model_dir / "model.pkl").resolve())
print("Scaler path:", (model_dir / "scaler.pkl").resolve())
print("Features count:", len(features))
