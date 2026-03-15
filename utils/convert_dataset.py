import numpy as np

print("Loading npz...")
data = np.load("data/reach_v3_demos.npz")

print("Saving as separate npy files...")
np.save("data/images.npy",       data["images"])    # stays uint8
np.save("data/states.npy",       data["states"])
np.save("data/actions.npy",      data["actions"])
np.save("data/episode_ends.npy", data["episode_ends"])
print("Done")