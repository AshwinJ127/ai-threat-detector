import os
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("LIGHTGBM TRAINING - HIGH RECALL OPTIMIZED (FAST + STABLE)")
print("=" * 80)

# ---------------------------------------------------------------------
# Verify dataset path
# ---------------------------------------------------------------------
data_dir = "data/processed"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' not found. "
                            "Please ensure your preprocessed CSVs are saved there.")

# ---------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------
print("\nLoading preprocessed data...")
X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["Label"]
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["Label"]

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(label_encoder.classes_)}")

# ---------------------------------------------------------------------
# Downsample BENIGN class to balance data
# ---------------------------------------------------------------------
print("\nDownsampling BENIGN class for balance...")
train_df = X_train.copy()
train_df["Label"] = y_train

benign = train_df[train_df["Label"] == "BENIGN"].sample(frac=0.15, random_state=42)
attack = train_df[train_df["Label"] != "BENIGN"]
train_balanced = pd.concat([benign, attack])

X_train_bal = train_balanced.drop("Label", axis=1)
y_train_bal = train_balanced["Label"]

print(f"New training size: {len(X_train_bal):,}")
print(f"BENIGN ratio reduced to {len(benign) / len(train_balanced):.2%}")

# ---------------------------------------------------------------------
# Determine device
# ---------------------------------------------------------------------
try:
    device = "gpu" if lgb.has_gpu() and not os.uname().sysname.lower().startswith("darwin") else "cpu"
except Exception:
    device = "cpu"

print(f"\nUsing device: {device.upper()}")

# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
model = lgb.LGBMClassifier(
    device=device,
    boosting_type="gbdt",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.3,
    reg_lambda=0.5,
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    force_row_wise=True
)

# ---------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------
print("\nStarting LightGBM training with early stopping...")
start = time.time()

evals_result = {}
model.fit(
    X_train_bal, y_train_bal,
    eval_set=[(X_test, y_test)],
    eval_metric="multi_logloss",
    callbacks=[
        lgb.early_stopping(50, verbose=True),
        lgb.log_evaluation(50),
        lgb.record_evaluation(evals_result)
    ]
)

train_time = time.time() - start
print(f"\nTraining completed in {train_time/60:.2f} minutes")

# ---------------------------------------------------------------------
# Save model and checkpoint
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/lightgbm_model_{timestamp}.pkl"
checkpoint_path = f"models/lightgbm_checkpoint_{timestamp}.txt"

os.makedirs("models", exist_ok=True)
model.booster_.save_model(checkpoint_path)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Saved model: {model_path}")
print(f"Saved checkpoint: {checkpoint_path}")

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
print("\nEvaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)
print("\nDetailed Classification Report:\n", report)

# ---------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------
print("\nGenerating confusion matrix plot...")
cm = confusion_matrix(y_test, y_pred)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("LightGBM - Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"results/lightgbm_confusion_matrix_{timestamp}.png", dpi=300, bbox_inches="tight")
# ---------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------
print("\nExtracting feature importances...")
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

os.makedirs("results", exist_ok=True)
feature_importance.to_csv(f"results/lightgbm_feature_importance_{timestamp}.csv", index=False)

print("\n" + "=" * 80)
print("LIGHTGBM HIGH-RECALL TRAINING COMPLETE")
print("=" * 80)
