import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load combined dataset
data_path = "data/combined_dataset.csv"
df = pd.read_csv(data_path, low_memory=False)
print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]:,} columns\n")

# Clean up column names
df.columns = df.columns.str.strip()

# ---------------------------------------------------------
# 1. Quick Overview
# ---------------------------------------------------------
print("Dataset Preview:")
print(df.head(3))
print("\nColumns:")
print(df.columns.tolist())
print("\n")

# ---------------------------------------------------------
# 2. Missing Values Check
# ---------------------------------------------------------
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if not missing_cols.empty:
    print("Columns with missing values:")
    print(missing_cols.sort_values(ascending=False))
else:
    print("No missing values detected.")
print("\n")

# ---------------------------------------------------------
# 3. Label Distribution
# ---------------------------------------------------------
if "Label" not in df.columns:
    raise ValueError("'Label' column not found. Please check your dataset headers.")

print("Label Distribution:")
print(df["Label"].value_counts())
print("\n")

df["is_attack"] = np.where(df["Label"] == "BENIGN", 0, 1)
attack_ratio = df["is_attack"].mean()
print(f"Attack ratio: {attack_ratio:.2%} of total traffic\n")

# ---------------------------------------------------------
# 4. Feature Types and Cleaning
# ---------------------------------------------------------
drop_cols = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp", 
    "day", "source_file"
]
drop_cols = [c for c in drop_cols if c in df.columns]
df_clean = df.drop(columns=drop_cols, errors="ignore")

print(f"Dropped {len(drop_cols)} non-feature columns. Remaining features: {df_clean.shape[1]}\n")

cat_cols = df_clean.select_dtypes(include=["object"]).columns
if len(cat_cols) > 0:
    print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols.tolist()}")
    df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
else:
    print("No categorical columns to encode.")
print("\n")

# ---------------------------------------------------------
# 5. Normalization with inf handling
# ---------------------------------------------------------
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

# Replace infinities with NaN
df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
print(f"Normalized {len(numeric_cols)} numeric columns.\n")

# ---------------------------------------------------------
# 6. Train/Test Split
# ---------------------------------------------------------
X = df_clean.drop(columns=["is_attack"])
y = df_clean["is_attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]:,} samples")
print(f"Test set:  {X_test.shape[0]:,} samples\n")

# ---------------------------------------------------------
# 7. Save Preprocessed Data
# ---------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Saved preprocessed train/test data to data/processed/")
