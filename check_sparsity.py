import os
import numpy as np
from tqdm import tqdm

TRAIN_MEMMAP_PATH = "encoded_train_features.mmap"

print("Checking sparsity...")
num_train = len([f for f in os.listdir("encoded_train_npz") if f.endswith(".npz")])
X_train_raw = np.memmap(TRAIN_MEMMAP_PATH, dtype='float32', mode='r')
X_train = X_train_raw.reshape(num_train, -1)
n_features = X_train.shape[1]

# Check first 500 samples
sample_size = min(500, num_train)
X_sample = np.array(X_train[:sample_size])

non_zero_cols = np.any(X_sample != 0, axis=0)
nnz_ratio = np.sum(non_zero_cols) / n_features
print(f"Non-zero columns in first {sample_size} samples: {np.sum(non_zero_cols):,} / {n_features:,} ({nnz_ratio*100:.2f}%)")

global_non_zero = np.zeros(n_features, dtype=bool)
for i in tqdm(range(0, num_train, 128)):
    batch = np.array(X_train[i:i+128])
    global_non_zero |= np.any(batch != 0, axis=0)

global_nnz_ratio = np.sum(global_non_zero) / n_features
print(f"Global non-zero columns: {np.sum(global_non_zero):,} / {n_features:,} ({global_nnz_ratio*100:.2f}%)")
