import pandas as pd
from sklearn.model_selection import train_test_split
from ml_ids.transform.preprocessing import remove_inf_values, remove_negative_values
import os

# Input and output paths
input_csv = "data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

print("🔹 Loading dataset...")
df = pd.read_csv(input_csv)
print("✅ Loaded", df.shape, "rows")

print("🔹 Cleaning dataset...")
df = remove_inf_values(df)
df = remove_negative_values(df)
print("✅ Cleaned")

print("🔹 Splitting dataset (train/val/test)...")
train, test = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

train.to_hdf(os.path.join(output_dir, "train.h5"), key="data", mode="w")
val.to_hdf(os.path.join(output_dir, "val.h5"), key="data", mode="w")
test.to_hdf(os.path.join(output_dir, "test.h5"), key="data", mode="w")

print("✅ Saved:")
print("  data/processed/train.h5")
print("  data/processed/val.h5")
print("  data/processed/test.h5")
