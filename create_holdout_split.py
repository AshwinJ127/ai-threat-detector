import pandas as pd
import os
from sklearn.model_selection import train_test_split

data_dir = "data/processed"
os.makedirs(data_dir, exist_ok=True)

# Load your full preprocessed dataset
X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
X_test = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["Label"]
y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["Label"]

# Combine all for a global split
X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

# Split out 10% as holdout
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X_full, y_full, test_size=0.10, stratify=y_full, random_state=42
)

# Split remaining into train/test again
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=42
)

# Save all splits
X_train_new.to_csv(os.path.join(data_dir, "X_train_scaled.csv"), index=False)
X_test_new.to_csv(os.path.join(data_dir, "X_test_scaled.csv"), index=False)
y_train_new.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
y_test_new.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)
X_holdout.to_csv(os.path.join(data_dir, "X_holdout_scaled.csv"), index=False)
y_holdout.to_csv(os.path.join(data_dir, "y_holdout.csv"), index=False)

print(f"Holdout set created and saved!  Holdout size: {len(X_holdout):,}")
