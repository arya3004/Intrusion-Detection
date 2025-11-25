import h5py

file_path = r".\data\test.h5"

with h5py.File(file_path, "r") as f:
    print("Keys inside test.h5:")
    for key in f.keys():
        print("  →", key)
