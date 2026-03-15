import numpy as np
from envs.metaworld_env import MetaWorldEnv
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

TASK           = "reach-v3"
EPISODES       = 100
MAX_STEPS      = 200
IMG_SIZE       = 224
ACTION_HORIZON = 16           # chunk size
SAVE_PATH      = "data/reach_v3_demos.npz"


def main():
    env    = MetaWorldEnv(TASK, img_size=IMG_SIZE)
    policy = SawyerReachV3Policy()

    all_images, all_states, all_actions, episode_ends = [], [], [], []

    for ep in range(EPISODES):
        obs  = env.reset()
        done = False
        step = 0

        ep_images  = []
        ep_states  = []
        ep_actions = []

        # collect full episode first
        while not done and step < MAX_STEPS:
            action = policy.get_action(obs["state"])
            ep_images.append(obs["image"])
            ep_states.append(obs["state"])
            ep_actions.append(action)
            obs, _, done, info = env.step(action)
            if int(info.get("success", 0)) == 1:
                done = True
            step += 1

        T = len(ep_actions)

        # build (obs, action_chunk) pairs for each timestep
        # at timestep t, target is actions[t : t+ACTION_HORIZON]
        # pad with last action if episode ends before horizon is full
        for t in range(T):
            chunk = ep_actions[t : t + ACTION_HORIZON]
            # pad if near end of episode
            while len(chunk) < ACTION_HORIZON:
                chunk.append(ep_actions[-1])

            all_images.append(ep_images[t])
            all_states.append(ep_states[t])
            all_actions.append(np.stack(chunk))   # (ACTION_HORIZON, action_dim)

        episode_ends.append(len(all_images) - 1)
        print(f"Episode {ep+1} | steps: {T}")

    print("Converting and saving...")
    np.savez_compressed(
        SAVE_PATH,
        images        = np.array(all_images,  dtype=np.uint8),
        states        = np.array(all_states,  dtype=np.float32),
        actions       = np.array(all_actions, dtype=np.float32),  # (N, H, action_dim)
        episode_ends  = np.array(episode_ends),
    )
    print("Saved:", SAVE_PATH)


if __name__ == "__main__":
    main()