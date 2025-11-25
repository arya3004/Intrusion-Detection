import pandas as pd
from pathlib import Path

raw_csv_path = Path("data/raw/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
out_dir = Path("data/processed")
out_dir.mkdir(exist_ok=True, parents=True)

# --- 1. Load CSV ---
df = pd.read_csv(raw_csv_path, header=0)  # adjust sep=',' or sep=';' if needed
print(f"Loaded shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# --- 2. Check for mixed-type columns ---
for col in df.columns:
    if df[col].apply(type).nunique() > 1:
        print(f"Mixed-type column: {col}, types: {df[col].apply(type).unique()}")

# --- 3. Optional: convert mixed columns to string ---
for col in df.select_dtypes(['object']):
    df[col] = df[col].astype(str)

# --- 4. Save to HDF5 ---
df.to_hdf(out_dir / "train.h5", key="data", mode="w", format="table")
