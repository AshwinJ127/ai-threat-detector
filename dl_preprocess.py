import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle

print("="*80)
print("DEEP LEARNING PREPROCESSING PIPELINE")
print("="*80)

print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv('data/combined_dataset.csv')
df.columns = df.columns.str.strip()

print(f"Initial shape: {df.shape}")

print("\n" + "="*80)
print("STEP 2: REMOVE USELESS COLUMNS")
print("="*80)

# Remove zero-variance columns
zero_variance_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Removing {len(zero_variance_cols)} zero-variance columns")
df = df.drop(columns=zero_variance_cols)

# Keep day and source_file for now (might help with sequence creation)
print(f"Shape after removing useless columns: {df.shape}")

print("\n" + "="*80)
print("STEP 3: HANDLE INVALID VALUES")
print("="*80)

# Fix negative durations
df['Flow Duration'] = df['Flow Duration'].abs()

# Replace infinite values with NaN, then fill
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

print("Cleaned negative and infinite values")

print("\n" + "="*80)
print("STEP 4: FEATURE ENGINEERING FOR SEQUENCES")
print("="*80)

# Sort by timestamp-related features to maintain temporal order
# Since we don't have explicit timestamps, we'll use row order as proxy
# In production, you'd sort by actual timestamps

# Add a sequence ID (for grouping flows into sequences)
# This is simplified - in practice, you'd group by IP pair + time window
print("Creating sequence groups...")
sequence_length = 10  # Number of flows per sequence
df['sequence_id'] = df.index // sequence_length

print(f"Created {df['sequence_id'].nunique()} sequences of length {sequence_length}")

print("\n" + "="*80)
print("STEP 5: SEPARATE FEATURES AND TARGET")
print("="*80)

# Separate features (X) and target (y)
# Keep sequence_id for later grouping
X = df.drop(columns=['Label'])
y = df['Label']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\n" + "="*80)
print("STEP 6: ENCODE TARGET LABELS")
print("="*80)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i}: {label}")

with open('label_encoder_dl.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\n" + "="*80)
print("STEP 7: TRAIN/TEST SPLIT (SEQUENCE-AWARE)")
print("="*80)

# Split by sequence_id to avoid data leakage
unique_sequences = df['sequence_id'].unique()
train_seq, test_seq = train_test_split(
    unique_sequences, 
    test_size=0.2, 
    random_state=42
)

train_mask = df['sequence_id'].isin(train_seq)
test_mask = df['sequence_id'].isin(test_seq)

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y_encoded[train_mask]
y_test = y_encoded[test_mask]

print(f"Training sequences: {len(train_seq)}")
print(f"Test sequences: {len(test_seq)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\n" + "="*80)
print("STEP 8: FEATURE SCALING (MinMaxScaler for DL)")
print("="*80)

# Use MinMaxScaler for neural networks (scales to 0-1 range)
# Exclude metadata columns from scaling
metadata_cols = ['day', 'source_file', 'sequence_id']
feature_cols = [col for col in X.columns if col not in metadata_cols]

scaler = MinMaxScaler()
X_train_numerical = X_train[feature_cols]
X_test_numerical = X_test[feature_cols]

X_train_scaled = scaler.fit_transform(X_train_numerical)
X_test_scaled = scaler.transform(X_test_numerical)

print(f"Scaled {len(feature_cols)} numerical features to [0, 1] range")

# Convert back to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

# Add back metadata columns
for col in metadata_cols:
    if col in X_train.columns:
        X_train_scaled_df[col] = X_train[col].values
        X_test_scaled_df[col] = X_test[col].values

with open('scaler_dl.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*80)
print("STEP 9: CREATE SEQUENCES FOR TRANSFORMER")
print("="*80)

def create_sequences(X_df, y_arr, seq_length=10):
    """
    Create sequences of flows for transformer input
    Returns: (sequences, labels)
    - sequences: shape (num_sequences, seq_length, num_features)
    - labels: shape (num_sequences,) - label for each sequence
    """
    sequences = []
    labels = []
    
    # Group by sequence_id
    for seq_id in X_df['sequence_id'].unique():
        seq_mask = X_df['sequence_id'] == seq_id
        seq_data = X_df[seq_mask][feature_cols].values
        seq_labels = y_arr[seq_mask]
        
        # Only keep sequences of exact length
        if len(seq_data) == seq_length:
            sequences.append(seq_data)
            # Use majority label for the sequence (or last label)
            labels.append(np.bincount(seq_labels).argmax())
    
    return np.array(sequences), np.array(labels)

print("Creating sequences for training set...")
X_train_seq, y_train_seq = create_sequences(X_train_scaled_df, y_train, sequence_length)

print("Creating sequences for test set...")
X_test_seq, y_test_seq = create_sequences(X_test_scaled_df, y_test, sequence_length)

print(f"\nSequence shapes:")
print(f"  X_train_seq: {X_train_seq.shape} (num_sequences, seq_length, num_features)")
print(f"  y_train_seq: {y_train_seq.shape}")
print(f"  X_test_seq: {X_test_seq.shape}")
print(f"  y_test_seq: {y_test_seq.shape}")

print("\n" + "="*80)
print("STEP 10: SAVE PROCESSED DATA")
print("="*80)

# Save sequences as numpy arrays (better for DL)
np.save('data/processed/X_train_seq.npy', X_train_seq)
np.save('data/processed/y_train_seq.npy', y_train_seq)
np.save('data/processed/X_test_seq.npy', X_test_seq)
np.save('data/processed/y_test_seq.npy', y_test_seq)

# Also save flat versions (for baseline models)
X_train_scaled_df[feature_cols].to_csv('data/processed/X_train_dl.csv', index=False)
X_test_scaled_df[feature_cols].to_csv('data/processed/X_test_dl.csv', index=False)
pd.DataFrame(y_train, columns=['Label']).to_csv('data/processed/y_train_dl.csv', index=False)
pd.DataFrame(y_test, columns=['Label']).to_csv('data/processed/y_test_dl.csv', index=False)

print("Saved processed data:")
print("  Sequence data (for transformers):")
print("    - data/processed/X_train_seq.npy")
print("    - data/processed/y_train_seq.npy")
print("    - data/processed/X_test_seq.npy")
print("    - data/processed/y_test_seq.npy")
print("  Flat data (for baseline models):")
print("    - data/processed/X_train_dl.csv")
print("    - data/processed/X_test_dl.csv")
print("    - data/processed/y_train_dl.csv")
print("    - data/processed/y_test_dl.csv")

print("\n" + "="*80)
print("STEP 11: DATA SUMMARY FOR MODEL ARCHITECTURE")
print("="*80)

print(f"\nModel input specifications:")
print(f"  Sequence length: {sequence_length}")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Number of classes: {len(label_encoder.classes_)}")
print(f"  Training sequences: {len(X_train_seq):,}")
print(f"  Test sequences: {len(X_test_seq):,}")
print(f"  Input shape for transformer: (batch_size, {sequence_length}, {len(feature_cols)})")

print("\nClass distribution in sequences:")
for i, label in enumerate(label_encoder.classes_):
    train_count = (y_train_seq == i).sum()
    test_count = (y_test_seq == i).sum()
    print(f"  {label}: Train={train_count}, Test={test_count}")

print("\n" + "="*80)
print("DL PREPROCESSING COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Build transformer architecture")
print("  2. Handle class imbalance (class weights or SMOTE)")
print("  3. Train model with early stopping")
print("  4. Evaluate and tune hyperparameters")