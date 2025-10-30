# src/create_dummy_model.py
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)

# Tiny fake dataset
X = np.random.rand(200, 5)
y = (X[:,0] + X[:,1]*0.3 > 0.9).astype(int)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(Xs, y)

joblib.dump(model, model_dir / "model.pkl")
joblib.dump(scaler, model_dir / "scaler.pkl")

print("Dummy model and scaler created at:", model_dir.resolve())
print("Model size (bytes):", (model_dir / "model.pkl").stat().st_size)
