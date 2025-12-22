import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import time

print("="*80)
print("BASELINE MODEL TRAINING - AI THREAT DETECTOR")
print("="*80)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')['Label']
y_test = pd.read_csv('data/processed/y_test.csv')['Label']

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(label_encoder.classes_)}")

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("\nClass weights (to handle imbalance):")
for class_idx, weight in class_weight_dict.items():
    print(f"  {label_encoder.classes_[class_idx]}: {weight:.2f}")

print("\n" + "="*80)
print("MODEL 1: RANDOM FOREST")
print("="*80)

# Train Random Forest
print("\nTraining Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time

print(f"Training completed in {rf_train_time:.2f} seconds")

# Predictions
print("\nMaking predictions...")
start_time = time.time()
y_pred_rf = rf_model.predict(X_test)
rf_pred_time = time.time() - start_time

# Evaluation
print("\n" + "="*80)
print("RANDOM FOREST RESULTS")
print("="*80)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"\nOverall Accuracy: {rf_accuracy:.4f}")
print(f"Prediction time: {rf_pred_time:.2f} seconds")
print(f"Predictions per second: {len(y_test)/rf_pred_time:.0f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, 
                          target_names=label_encoder.classes_,
                          digits=4))

# Feature Importance
print("\nTop 20 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('rf_feature_importance.csv', index=False)

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved confusion matrix to 'rf_confusion_matrix.png'")

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Saved model to 'rf_model.pkl'")

print("\n" + "="*80)
print("MODEL 2: XGBOOST")
print("="*80)

# Train XGBoost
print("\nTraining XGBoost...")
start_time = time.time()

# Convert class weights to sample weights
sample_weights = np.array([class_weight_dict[y] for y in y_train])

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    verbosity=1
)

xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
xgb_train_time = time.time() - start_time

print(f"Training completed in {xgb_train_time:.2f} seconds")

# Predictions
print("\nMaking predictions...")
start_time = time.time()
y_pred_xgb = xgb_model.predict(X_test)
xgb_pred_time = time.time() - start_time

# Evaluation
print("\n" + "="*80)
print("XGBOOST RESULTS")
print("="*80)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"\nOverall Accuracy: {xgb_accuracy:.4f}")
print(f"Prediction time: {xgb_pred_time:.2f} seconds")
print(f"Predictions per second: {len(y_test)/xgb_pred_time:.0f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, 
                          target_names=label_encoder.classes_,
                          digits=4))

# Feature Importance
print("\nTop 20 Most Important Features:")
xgb_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in xgb_feature_importance.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

xgb_feature_importance.to_csv('xgb_feature_importance.csv', index=False)

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(14, 12))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('XGBoost - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nSaved confusion matrix to 'xgb_confusion_matrix.png'")

# Save model
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("Saved model to 'xgb_model.pkl'")

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_accuracy, xgb_accuracy],
    'Training Time (s)': [rf_train_time, xgb_train_time],
    'Prediction Time (s)': [rf_pred_time, xgb_pred_time],
    'Predictions/sec': [len(y_test)/rf_pred_time, len(y_test)/xgb_pred_time]
})

print("\n", comparison.to_string(index=False))

# Per-class performance comparison
print("\n" + "="*80)
print("PER-CLASS PERFORMANCE")
print("="*80)

rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_rf, average=None)
xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(
    y_test, y_pred_xgb, average=None)

per_class_perf = pd.DataFrame({
    'Class': label_encoder.classes_,
    'RF_Precision': rf_precision,
    'XGB_Precision': xgb_precision,
    'RF_Recall': rf_recall,
    'XGB_Recall': xgb_recall,
    'RF_F1': rf_f1,
    'XGB_F1': xgb_f1
})

print("\n", per_class_perf.to_string(index=False))
per_class_perf.to_csv('per_class_performance.csv', index=False)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - rf_model.pkl")
print("  - xgb_model.pkl")
print("  - rf_confusion_matrix.png")
print("  - xgb_confusion_matrix.png")
print("  - rf_feature_importance.csv")
print("  - xgb_feature_importance.csv")
print("  - per_class_performance.csv")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review confusion matrices to identify problematic classes")
print("2. Analyze feature importance for insights")
print("3. Consider feature selection based on importance")
print("4. Address class imbalance with advanced techniques if needed")
print("5. Build transformer model to compare with baseline")
print("6. Deploy best model in production pipeline")