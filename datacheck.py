import os
import pandas as pd
import numpy as np
from collections import Counter

DATA_DIR = "data/processed"

# Load data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Label"]
X_val = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))  # or your validation split
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["Label"]

print("=== Dataset Shape ===")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}\n")

# Class distribution
print("=== Class Distribution ===")
train_counts = Counter(y_train)
val_counts = Counter(y_val)
print("Train:", train_counts)
print("Validation:", val_counts, "\n")

# Percent of missing values per column
print("=== Missing Values ===")
missing_train = X_train.isnull().mean() * 100
missing_val = X_val.isnull().mean() * 100
print("Train missing (%):")
print(missing_train[missing_train > 0])
print("Validation missing (%):")
print(missing_val[missing_val > 0], "\n")

# Basic stats per feature
print("=== Feature Stats ===")
feature_stats = X_train.describe().T
feature_stats['std/mean'] = feature_stats['std'] / (feature_stats['mean'] + 1e-8)
print(feature_stats[['mean', 'std', 'std/mean']].head(10))  # top 10 features for brevity

# Check for identical features or near-constant features
constant_features = [col for col in X_train.columns if X_train[col].nunique() <= 1]
print("Constant features:", constant_features)

# Check for features that are identical in all rows
duplicated_features = X_train.columns[X_train.T.duplicated()].tolist()
print("Duplicated features:", duplicated_features, "\n")

# Correlation between features
corr_matrix = X_train.corr().abs()
high_corr = np.where((corr_matrix > 0.95) & (corr_matrix < 1.0))
high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr)]
print(f"Highly correlated feature pairs (>0.95): {high_corr_pairs[:10]} (showing first 10)\n")

# Optional: Check train/val label coverage
missing_classes_in_val = set(y_train.unique()) - set(y_val.unique())
print("Classes present in train but missing in validation:", missing_classes_in_val)
