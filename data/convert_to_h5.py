import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your CSV file
csv_path = "data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

print("📥 Loading CSV file...")
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

# Split into train, val, and test sets (70/15/15)
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save each as .h5
train.to_hdf("data/train.h5", key="data", mode="w")
val.to_hdf("data/val.h5", key="data", mode="w")
test.to_hdf("data/test.h5", key="data", mode="w")

print("🎯 Conversion complete! Files saved in 'data/' as:")
print("   - train.h5")
print("   - val.h5")
print("   - test.h5")
