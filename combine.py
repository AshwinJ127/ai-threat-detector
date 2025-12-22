import pandas as pd
import os
import glob

# Path to your data folder
data_dir = "data"

# Get all CSVs in the folder
files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

print("Found the following files to merge:\n")
for f in files:
    print(f"  • {os.path.basename(f)}")
print("\n------------------------------------------\n")

dfs = []
total_rows = 0

for f in files:
    try:
        df = pd.read_csv(f, low_memory=False)
        num_rows, num_cols = df.shape
        total_rows += num_rows

        # Extract weekday label from filename (e.g., "Monday", "Friday", etc.)
        day_label = os.path.basename(f).split("-")[0]
        df["day"] = day_label

        # Add an identifier for which file this came from
        df["source_file"] = os.path.basename(f)

        print(f"Loaded {os.path.basename(f)} — {num_rows:,} rows × {num_cols:,} cols")

        dfs.append(df)

    except Exception as e:
        print(f"Error reading {f}: {e}")

print("\n------------------------------------------")
print(f"Successfully loaded {len(dfs)} files with a total of {total_rows:,} rows.")
print("------------------------------------------\n")

# Combine everything
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape[0]:,} rows × {combined_df.shape[1]:,} columns")

# Show how many rows came from each file
print("\nRows per source file:")
print(combined_df['source_file'].value_counts())

# Show how many rows per day label
print("\nRows per day of week:")
print(combined_df['day'].value_counts())

# Save the merged CSV
output_path = os.path.join(data_dir, "combined_dataset.csv")
combined_df.to_csv(output_path, index=False)
print(f"\nSaved combined dataset → {output_path}")
