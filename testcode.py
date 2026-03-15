import numpy as np
data = np.load("data/reach_v3_demos.npz")
print(data["actions"].shape)
print(data["images"].shape)
print(data["states"].shape)

print(len(data["episode_ends"]))         # should be 500
print(data["episode_ends"][-1])          # should be total timesteps - 1

import sys
print(f"images uncompressed:  {data['images'].nbytes  / 1024**3:.2f} GB")
print(f"states uncompressed:  {data['states'].nbytes  / 1024**2:.1f} MB")
print(f"actions uncompressed: {data['actions'].nbytes / 1024**2:.1f} MB")