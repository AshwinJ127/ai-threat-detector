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
    confusion_matrix,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("LIGHTGBM TRAINING - HIGH RECALL OPTIMIZED (SAFE + STABLE)")
print("=" * 80)

# ---------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------
DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------
print("\nLoading preprocessed data...")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Label"]
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["Label"]

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

NUM_CLASSES = len(label_encoder.classes_)
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {NUM_CLASSES}")

# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------
print("\nComputing class weights...")
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Class weights: {class_weight_dict}")

# ---------------------------------------------------------------------
# Stratified validation split
# ---------------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)
print(f"Training set: {len(X_tr):,}, Validation set: {len(X_val):,}")

# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
try:
    device = "gpu" if lgb.has_gpu() and not os.uname().sysname.lower().startswith("darwin") else "cpu"
except Exception:
    device = "cpu"
print(f"Using device: {device.upper()}")

# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
model = lgb.LGBMClassifier(
    device=device,
    boosting_type="gbdt",
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=128,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=0.5,
    objective="multiclass",
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1,
    force_row_wise=True
)

# ---------------------------------------------------------------------
# Custom metric: macro recall (for scikit-learn API)
# ---------------------------------------------------------------------
def macro_recall(y_true, y_pred):
    y_pred_classes = y_pred.reshape(-1, NUM_CLASSES).argmax(axis=1)
    recall = recall_score(y_true, y_pred_classes, average="macro")
    return "macro_recall", recall, True

# ---------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------
print("\nStarting LightGBM training with early stopping...")
start = time.time()
evals_result = {}

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric=macro_recall,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
        lgb.record_evaluation(evals_result)
    ]
)
train_time = time.time() - start
print(f"\nTraining completed in {train_time/60:.2f} minutes")

# ---------------------------------------------------------------------
# Save outputs with timestamp
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = os.path.join(MODELS_DIR, f"lightgbm_model_{timestamp}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

booster_path = os.path.join(MODELS_DIR, f"lightgbm_booster_{timestamp}.txt")
model.booster_.save_model(booster_path, num_iteration=model.best_iteration_)

print(f"Saved model: {model_path}")
print(f"Saved booster: {booster_path}")

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
print("\nEvaluating model on test set...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)
print("\nDetailed Classification Report:\n", report)

# ---------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
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
cm_path = os.path.join(RESULTS_DIR, f"lightgbm_confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix: {cm_path}")

# ---------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fi_path = os.path.join(RESULTS_DIR, f"lightgbm_feature_importance_{timestamp}.csv")
feature_importance.to_csv(fi_path, index=False)
print(f"Saved feature importance: {fi_path}")

print("\n" + "=" * 80)
print("LIGHTGBM HIGH-RECALL TRAINING COMPLETE (SAFE)")
print("=" * 80)
