import numpy as np
from envs.metaworld_env import MetaWorldEnv
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy

TASK           = "reach-v3"
EPISODES       = 1000
MAX_STEPS      = 200
IMG_SIZE       = 224
ACTION_HORIZON = 8
OBS_HORIZON    = 2
SAVE_DIR       = "data"
CHUNK_SIZE     = 64     # decrease this if your pc lags saving a chunl 
 # save every N episodes, then clear RAM. 


def collect_chunk(env, policy, n_episodes, start_ep):
    all_images, all_states, all_actions, episode_ends = [], [], [], []

    for ep in range(n_episodes):
        obs  = env.reset()
        done = False
        step = 0

        ep_images, ep_states, ep_actions = [], [], []

        while not done and step < MAX_STEPS:
            action = policy.get_action(obs["state"])
            ep_images.append(obs["image"])
            ep_states.append(obs["state"])
            ep_actions.append(action)
            obs, _, done, info = env.step(action)
            if int(info.get("success", 0)) == 1:
                ep_images.append(obs["image"])
                ep_states.append(obs["state"])
                ep_actions.append(np.zeros_like(action))
                done = True
            step += 1

        T = len(ep_actions)
        for t in range(T):
            chunk = ep_actions[t : t + ACTION_HORIZON]
            while len(chunk) < ACTION_HORIZON:
                chunk.append(ep_actions[-1])
            all_images.append(ep_images[t])
            all_states.append(ep_states[t])
            all_actions.append(np.stack(chunk))

        episode_ends.append(len(all_images) - 1)
        print(f"Episode {start_ep + ep + 1} | steps: {T}")

    return (
        np.array(all_images,  dtype=np.uint8),
        np.array(all_states,  dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        np.array(episode_ends),
    )


def append_to_npy(path, new_data):
    """Append new_data to existing .npy file, or create it if it doesn't exist."""
    try:
        existing = np.load(path, mmap_mode="r")
        combined = np.concatenate([existing, new_data], axis=0)
    except FileNotFoundError:
        combined = new_data
    np.save(path, combined)


def main():
    env    = MetaWorldEnv(TASK, img_size=IMG_SIZE, obs_horizon=OBS_HORIZON)
    policy = SawyerReachV3Policy()

    n_chunks = EPISODES // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        start_ep = chunk_idx * CHUNK_SIZE
        print(f"\n── Chunk {chunk_idx+1}/{n_chunks} (episodes {start_ep+1}-{start_ep+CHUNK_SIZE}) ──")

        images, states, actions, episode_ends = collect_chunk(
            env, policy, CHUNK_SIZE, start_ep
        )

        print("Saving chunk to disk...")
        append_to_npy(f"{SAVE_DIR}/images.npy",       images)
        append_to_npy(f"{SAVE_DIR}/states.npy",        states)
        append_to_npy(f"{SAVE_DIR}/actions.npy",       actions)
        append_to_npy(f"{SAVE_DIR}/episode_ends.npy",  episode_ends)

        # explicitly free RAM before next chunk
        del images, states, actions, episode_ends
        print(f"Chunk {chunk_idx+1} saved. RAM freed.")

    print(f"\nDone. Total samples: {np.load(f'{SAVE_DIR}/images.npy', mmap_mode='r').shape[0]}")


if __name__ == "__main__":
    main()
    