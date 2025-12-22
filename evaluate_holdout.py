import pickle
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from datetime import datetime

data_dir = "data/processed"
holdout_X = pd.read_csv(os.path.join(data_dir, "X_holdout_scaled.csv"))
holdout_y = pd.read_csv(os.path.join(data_dir, "y_holdout.csv"))["Label"]

# Load label encoder and model
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model_path = "models/lightgbm_model_20251112_182516.pkl"  
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Evaluate
print(f"🧪 Evaluating model '{model_path}' on holdout data...")
y_pred = model.predict(holdout_X)

acc = accuracy_score(holdout_y, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(holdout_y, y_pred, average="macro")

print(f"\nHoldout Accuracy: {acc:.4f}")
print(f"Holdout Precision: {prec:.4f}")
print(f"Holdout Recall: {rec:.4f}")
print(f"Holdout F1 Score: {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(holdout_y, y_pred, target_names=label_encoder.classes_, digits=4))
