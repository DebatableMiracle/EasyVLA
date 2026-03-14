import numpy as np
from envs.metaworld_env import MetaWorldEnv
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

TASK = "reach-v3"
EPISODES = 100
MAX_STEPS = 200
IMG_SIZE = 224        # drop to 84 or 128 if you want sub-500MB
SAVE_PATH = "data/reach_v3_demos.npz"

def main():
    env = MetaWorldEnv(TASK, img_size=IMG_SIZE)
    policy = SawyerReachV3Policy()

    all_images, all_states, all_actions, episode_ends = [], [], [], []

    for ep in range(EPISODES):
        obs = env.reset()
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action = policy.get_action(obs["state"])

            all_images.append(obs["image"])
            all_states.append(obs["state"])
            all_actions.append(action)

            obs, _, done, info = env.step(action)

            if int(info.get("success", 0)) == 1:
                all_images.append(obs["image"])
                all_states.append(obs["state"])
                all_actions.append(np.zeros_like(action))
                done = True

            step += 1

        episode_ends.append(len(all_images) - 1)
        print(f"Episode {ep} | steps: {step}")

    print("Converting and saving...")
    np.savez_compressed(
        SAVE_PATH,
        images=np.array(all_images, dtype=np.uint8),    # uint8 not float — 4x smaller
        states=np.array(all_states, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        episode_ends=np.array(episode_ends),
    )
    print("Saved:", SAVE_PATH)

if __name__ == "__main__":
    main()