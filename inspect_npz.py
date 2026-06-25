import numpy as np
import os

d = "/k_data1/i055ueno/reserch/encoded_train_npz"
if not os.path.exists(d):
    print(f"Directory {d} not found.")
else:
    f = os.listdir(d)[0]
    data = np.load(os.path.join(d, f), allow_pickle=True)
    label = data["label"]
    print(f"Type: {type(label)}")
    print(f"Shape: {getattr(label, 'shape', 'No shape')}")
    print(f"Content: {label}")
