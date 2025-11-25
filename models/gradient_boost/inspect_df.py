import h5py

file_path = r".\data\test.h5"

with h5py.File(file_path, "r") as f:
    print("Keys inside file:", list(f.keys()))
    df = f["df"][:]

print("\nType:", type(df))
print("Shape:", df.shape)
print("Sample entries:")
for i, row in enumerate(df[:5]):
    print(f"Row {i}: {row}")
