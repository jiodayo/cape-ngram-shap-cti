import numpy as np
import scipy.linalg

M = 512
N = 3440640
print("Allocating X...")
X = np.random.randn(M, N).astype(np.float32)

print("Testing scipy.linalg.svd with gesdd...")
try:
    scipy.linalg.svd(X, full_matrices=False, check_finite=False)
    print("scipy gesdd worked!")
except Exception as e:
    print(f"scipy gesdd failed: {e}")

print("Testing scipy.linalg.svd with gesvd...")
try:
    scipy.linalg.svd(X, full_matrices=False, check_finite=False, lapack_driver='gesvd')
    print("scipy gesvd worked!")
except Exception as e:
    print(f"scipy gesvd failed: {e}")

print("Testing numpy.linalg.svd...")
try:
    np.linalg.svd(X, full_matrices=False)
    print("numpy worked!")
except Exception as e:
    print(f"numpy failed: {e}")
