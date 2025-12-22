import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# Load dataset
df = pd.read_csv('data/combined_dataset.csv')
df.columns = df.columns.str.strip()  # Strip whitespace from column names

print(f"Initial shape: {df.shape}")
print(f"Initial memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*80)
print("STEP 2: REMOVE USELESS COLUMNS")
print("="*80)

# Remove columns with only one unique value
zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Removing {len(zero_variance_cols)} zero-variance columns:")
for col in zero_variance_cols:
    print(f"  - {col}")

df = df.drop(columns=zero_variance_cols)

# Remove metadata columns (optional - keep for analysis later)
metadata_cols = ['day', 'source_file']
print(f"\nRemoving metadata columns: {metadata_cols}")
df = df.drop(columns=metadata_cols)

print(f"Shape after removing useless columns: {df.shape}")

print("\n" + "="*80)
print("STEP 3: HANDLE NEGATIVE AND INVALID VALUES")
print("="*80)

# Handle negative flow durations
negative_duration_count = (df['Flow Duration'] < 0).sum()
print(f"Negative flow durations found: {negative_duration_count}")
print("Replacing negative durations with absolute values...")
df['Flow Duration'] = df['Flow Duration'].abs()

# Handle infinite values
print("\nHandling infinite values...")
inf_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        inf_count = np.isinf(df[col]).sum()
        inf_cols.append((col, inf_count))
        
if inf_cols:
    print("Columns with infinite values:")
    for col, count in inf_cols:
        print(f"  {col}: {count} infinite values")
        # Replace inf with NaN, then fill with median
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# Check for NaN values
print("\nChecking for NaN values...")
nan_counts = df.isnull().sum()
nan_cols = nan_counts[nan_counts > 0]
if len(nan_cols) > 0:
    print("Columns with NaN values:")
    print(nan_cols)
    # Fill NaN with median for numerical columns
    for col in nan_cols.index:
        if df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} with median: {median_val}")
else:
    print("No NaN values found!")

print(f"\nShape after cleaning: {df.shape}")

print("\n" + "="*80)
print("STEP 4: ANALYZE CLASS DISTRIBUTION")
print("="*80)

label_counts = df['Label'].value_counts()
print("\nClass distribution:")
print(label_counts)
print("\nPercentages:")
print((label_counts / len(df) * 100).round(2))

# Visualize class distribution
plt.figure(figsize=(12, 6))
label_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved class distribution plot to 'class_distribution.png'")

print("\n" + "="*80)
print("STEP 5: SEPARATE FEATURES AND TARGET")
print("="*80)

# Separate features (X) and target (y)
X = df.drop(columns=['Label'])
y = df['Label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")

print("\n" + "="*80)
print("STEP 6: ENCODE TARGET LABELS")
print("="*80)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")

# Save label encoder for later use
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("\nSaved label encoder to 'label_encoder.pkl'")

print("\n" + "="*80)
print("STEP 7: TRAIN/TEST SPLIT")
print("="*80)

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nTraining set class distribution:")
train_label_counts = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_label_counts.items():
    label_name = label_encoder.classes_[idx]
    print(f"  {label_name}: {count} ({count/len(y_train)*100:.2f}%)")

print("\n" + "="*80)
print("STEP 8: FEATURE SCALING")
print("="*80)

# Standardize features (important for many ML algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler (mean=0, std=1)")
print(f"Training set scaled shape: {X_train_scaled.shape}")
print(f"Test set scaled shape: {X_test_scaled.shape}")

# Save scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved scaler to 'scaler.pkl'")

print("\n" + "="*80)
print("STEP 9: SAVE PROCESSED DATA")
print("="*80)

# Convert back to DataFrame for easier handling
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Save to CSV
X_train_scaled_df.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv('data/processed/X_test_scaled.csv', index=False)
pd.DataFrame(y_train, columns=['Label']).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test, columns=['Label']).to_csv('data/processed/y_test.csv', index=False)

print("Saved processed data:")
print("  - data/processed/X_train_scaled.csv")
print("  - data/processed/X_test_scaled.csv")
print("  - data/processed/y_train.csv")
print("  - data/processed/y_test.csv")

print("\n" + "="*80)
print("STEP 10: FEATURE CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation with target (using a sample for speed)
sample_size = min(50000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train.iloc[sample_indices]
y_sample = y_train[sample_indices]

print(f"Analyzing feature correlations using sample of {sample_size} records...")

# Calculate correlation between each feature and target
correlations = []
for col in X_sample.columns:
    corr = np.corrcoef(X_sample[col], y_sample)[0, 1]
    correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\nTop 20 features by correlation with target:")
for i, (col, corr) in enumerate(correlations[:20], 1):
    print(f"  {i}. {col}: {corr:.4f}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE!")
print("="*80)
print(f"\nFinal dataset summary:")
print(f"  - Total samples: {len(df):,}")
print(f"  - Training samples: {len(X_train):,}")
print(f"  - Test samples: {len(X_test):,}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Classes: {len(label_encoder.classes_)}")
print(f"\nNext steps:")
print("  1. Train baseline model (Random Forest, XGBoost)")
print("  2. Evaluate model performance")
print("  3. Feature selection based on importance")
print("  4. Handle class imbalance if needed")
print("  5. Implement transformer model")