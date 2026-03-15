import numpy as np
actions = np.load("data/actions.npy", mmap_mode="r")
print(actions.shape)  # middle dim is your actual horizon