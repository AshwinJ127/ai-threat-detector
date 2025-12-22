import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/combined_dataset.csv')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*80)
print("LABEL DISTRIBUTION (Attack Types)")
print("="*80)
label_counts = df['Label'].value_counts()
print(label_counts)
print(f"\nPercentage distribution:")
print((label_counts / len(df) * 100).round(2))

print("\n" + "="*80)
print("DAYS DISTRIBUTION")
print("="*80)
print(df['day'].value_counts())

print("\n" + "="*80)
print("SOURCE FILES")
print("="*80)
print(df['source_file'].value_counts())

print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)
print(f"Destination Ports - Range: {df['Destination Port'].min()} to {df['Destination Port'].max()}")
print(f"Flow Duration - Range: {df['Flow Duration'].min()} to {df['Flow Duration'].max()}")
print(f"Total Packets (Fwd) - Range: {df['Total Fwd Packets'].min()} to {df['Total Fwd Packets'].max()}")
print(f"Total Packets (Bwd) - Range: {df['Total Backward Packets'].min()} to {df['Total Backward Packets'].max()}")

print("\n" + "="*80)
print("CHECKING FOR ANOMALIES")
print("="*80)
# Check for negative flow durations
negative_durations = (df['Flow Duration'] < 0).sum()
print(f"Negative Flow Durations: {negative_durations}")

# Check for infinite values
inf_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        inf_count = np.isinf(df[col]).sum()
        inf_cols.append((col, inf_count))
        
if inf_cols:
    print("\nColumns with Infinite Values:")
    for col, count in inf_cols:
        print(f"  {col}: {count} infinite values")
else:
    print("No infinite values found")

print("\n" + "="*80)
print("FEATURE GROUPS")
print("="*80)
print("\n1. Packet-related features:")
packet_features = [col for col in df.columns if 'Packet' in col]
for f in packet_features[:10]:  # Show first 10
    print(f"   - {f}")

print("\n2. Flag-related features:")
flag_features = [col for col in df.columns if 'Flag' in col]
for f in flag_features:
    print(f"   - {f}")

print("\n3. Timing (IAT) features:")
iat_features = [col for col in df.columns if 'IAT' in col]
for f in iat_features:
    print(f"   - {f}")

print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)
print("Columns with only 1 unique value (candidates for removal):")
single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
for col in single_value_cols:
    print(f"   - {col}: {df[col].unique()[0]}")

print("\n" + "="*80)
print("SAMPLE RECORDS BY ATTACK TYPE")
print("="*80)
# Show one example of each attack type
for label in df['Label'].unique()[:5]:  # First 5 labels
    print(f"\n{label}:")
    sample = df[df['Label'] == label].iloc[0]
    print(f"  Port: {sample['Destination Port']}, Duration: {sample['Flow Duration']}, "
          f"Fwd Packets: {sample['Total Fwd Packets']}, Bwd Packets: {sample['Total Backward Packets']}")