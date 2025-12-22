import os
import json
import pickle
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_NAME = "lightgbm_model_20251112_182516.pkl"  # <-- replace if needed
MODEL_DIR = "models"
REGISTRY_CSV = os.path.join(MODEL_DIR, "model_registry.csv")
METRICS_JSON = os.path.join(MODEL_DIR, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model_final.pkl")

# Latest known holdout metrics (from evaluate_holdout.py)
metrics = {
    "accuracy": 0.9989,
    "precision": 0.9213,
    "recall": 0.8858,
    "f1": 0.8945,
    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": MODEL_NAME,
    "final_model_path": FINAL_MODEL_PATH,
    "features": 70,
    "objective": "multiclass",
    "num_classes": 15,
    "notes": "High-recall optimized LightGBM model"
}

# ---------------------------------------------------------------------
# Verify model exists
# ---------------------------------------------------------------------
src_path = os.path.join(MODEL_DIR, MODEL_NAME)
if not os.path.exists(src_path):
    raise FileNotFoundError(f"Model '{src_path}' not found.")

# ---------------------------------------------------------------------
# Copy model as final version
# ---------------------------------------------------------------------
with open(src_path, "rb") as f_in, open(FINAL_MODEL_PATH, "wb") as f_out:
    f_out.write(f_in.read())

print(f"Promoted model to: {FINAL_MODEL_PATH}")

# ---------------------------------------------------------------------
# Save metrics JSON
# ---------------------------------------------------------------------
with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Saved metrics to: {METRICS_JSON}")

# ---------------------------------------------------------------------
# Update model registry CSV
# ---------------------------------------------------------------------
entry = pd.DataFrame([metrics])

if os.path.exists(REGISTRY_CSV):
    registry = pd.read_csv(REGISTRY_CSV)
    registry = pd.concat([registry, entry], ignore_index=True)
else:
    registry = entry

registry.to_csv(REGISTRY_CSV, index=False)
print(f"Updated model registry: {REGISTRY_CSV}")

print("\nModel finalization complete.")
