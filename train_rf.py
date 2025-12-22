import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

print("="*80)
print("IMPROVED RANDOM FOREST TRAINING - RECALL OPTIMIZED")
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

# Compute balanced class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

print("\nClass weights:")
for i, w in class_weight_dict.items():
    print(f"  {label_encoder.classes_[i]}: {w:.2f}")

# Model definition (more trees, shallower splits, slightly deeper depth)
rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,                    # allow full tree growth, better recall
    min_samples_split=4,               # more aggressive splits
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced_subsample', # random subset weighting per tree
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Cross-validation for stability check
print("\nRunning 3-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_results = cross_validate(
    rf_model, X_train, y_train,
    cv=cv,
    scoring=['recall_macro', 'f1_macro', 'accuracy'],
    n_jobs=-1,
    verbose=0
)

print(f"\nCV Mean Recall: {cv_results['test_recall_macro'].mean():.4f}")
print(f"CV Mean F1: {cv_results['test_f1_macro'].mean():.4f}")
print(f"CV Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")

print("\n" + "="*80)
print("FINAL TRAINING ON FULL DATA")
print("="*80)

start_time = time.time()
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"\nTraining completed in {train_time:.2f} seconds")

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)
print("\nClassification Report:\n", report)

# Save per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
per_class_perf = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})
print("\nPer-Class Metrics:")
print(per_class_perf.to_string(index=False))
per_class_perf.to_csv('rf_improved_performance.csv', index=False)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Improved Random Forest - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('rf_improved_confusion_matrix.png', dpi=300, bbox_inches='tight')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))
feature_importance.to_csv('rf_improved_feature_importance.csv', index=False)

# Save model
with open('rf_model_improved.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\nSaved improved model to 'rf_model_improved.pkl'")

print("\n" + "="*80)
print("IMPROVED RANDOM FOREST TRAINING COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - rf_model_improved.pkl")
print("  - rf_improved_confusion_matrix.png")
print("  - rf_improved_feature_importance.csv")
print("  - rf_improved_performance.csv")
