import numpy as np
import scipy.linalg

M = 512
N = 3440640

# Check scipy source logic for gesdd lwork
sz = M * N
print(f"M * N = {sz}")
