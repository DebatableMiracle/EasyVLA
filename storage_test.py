import numpy as np
import os

path = "data/drawer-close-v3/images.npy"   # or wherever your new data is
img  = np.load(path, mmap_mode="r")
print(f"shape:    {img.shape}")
print(f"dtype:    {img.dtype}")
print(f"size:     {img.nbytes / 1024**3:.2f} GB uncompressed")
print(f"on disk:  {os.path.getsize(path) / 1024**3:.2f} GB")
