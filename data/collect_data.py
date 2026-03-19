# data/collect_data.py
import numpy as np
import os
from envs.metaworld_env import MetaWorldEnv, TASK_CAMERAS
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerButtonPressTopdownV3Policy,
    SawyerPegInsertionSideV3Policy,
    SawyerWindowOpenV3Policy,
    SawyerWindowCloseV3Policy,
)

# ── choose your tasks here ────────────────────────────────────────────────────
TASKS = [
    "reach-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "door-open-v3",
    "push-v3",
]
# ─────────────────────────────────────────────────────────────────────────────

POLICY_MAP = {
    "reach-v3":                  SawyerReachV3Policy,
    "push-v3":                   SawyerPushV3Policy,
    "pick-place-v3":             SawyerPickPlaceV3Policy,
    "door-open-v3":              SawyerDoorOpenV3Policy,
    "drawer-close-v3":           SawyerDrawerCloseV3Policy,
    "drawer-open-v3":            SawyerDrawerOpenV3Policy,
    "button-press-topdown-v3":   SawyerButtonPressTopdownV3Policy,
    "peg-insert-side-v3":        SawyerPegInsertionSideV3Policy,
    "window-open-v3":            SawyerWindowOpenV3Policy,
    "window-close-v3":           SawyerWindowCloseV3Policy,
}

EPISODES_PER_TASK = 500
MAX_STEPS         = 200
IMG_SIZE          = 128
ACTION_HORIZON    = 8
OBS_HORIZON       = 3
CHUNK_SIZE        = 100
DATA_ROOT         = "data"


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
            chunk = ep_actions[t: t + ACTION_HORIZON]
            while len(chunk) < ACTION_HORIZON:
                chunk.append(ep_actions[-1])
            all_images.append(ep_images[t])
            all_states.append(ep_states[t])
            all_actions.append(np.stack(chunk))

        episode_ends.append(len(all_images) - 1)
        print(f"  ep {start_ep + ep + 1} | steps: {T}")

    return (
        np.array(all_images,  dtype=np.uint8),
        np.array(all_states,  dtype=np.float32),
        np.array(all_actions, dtype=np.float32),
        np.array(episode_ends),
    )


def append_npy(path, new_data):
    try:
        existing = np.load(path, mmap_mode="r")
        combined = np.concatenate([existing, new_data], axis=0)
    except FileNotFoundError:
        combined = new_data
    np.save(path, combined)


def collect_task(task_name):
    save_dir = os.path.join(DATA_ROOT, task_name)
    os.makedirs(save_dir, exist_ok=True)

    cam = TASK_CAMERAS.get(task_name, "corner2")
    print(f"\n{'='*50}")
    print(f"Collecting: {task_name} | camera: {cam}")
    print(f"Episodes: {EPISODES_PER_TASK} | IMG: {IMG_SIZE}x{IMG_SIZE} | OBS_H: {OBS_HORIZON}")
    print(f"{'='*50}")

    env    = MetaWorldEnv(
        task_name   = task_name,
        img_size    = IMG_SIZE,
        obs_horizon = OBS_HORIZON,
        camera_name = cam,          # always explicit, no None
    )
    policy = POLICY_MAP[task_name]()
    n_chunks = EPISODES_PER_TASK // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        start_ep = chunk_idx * CHUNK_SIZE
        print(f"\n── Chunk {chunk_idx+1}/{n_chunks} ──")

        images, states, actions, episode_ends = collect_chunk(
            env, policy, CHUNK_SIZE, start_ep
        )
        append_npy(os.path.join(save_dir, "images.npy"),      images)
        append_npy(os.path.join(save_dir, "states.npy"),       states)
        append_npy(os.path.join(save_dir, "actions.npy"),      actions)
        append_npy(os.path.join(save_dir, "episode_ends.npy"), episode_ends)

        del images, states, actions, episode_ends
        print(f"Chunk {chunk_idx+1} saved.")

    n = np.load(os.path.join(save_dir, "images.npy"), mmap_mode="r").shape[0]
    print(f"\n{task_name} done — {n} samples | camera: {cam} | {save_dir}/")
    env.close()


def main():
    print(f"Collecting {len(TASKS)} tasks: {TASKS}")
    for task in TASKS:
        if task not in POLICY_MAP:
            print(f"WARNING: no policy for {task}, skipping")
            continue
        collect_task(task)
    print("\nAll tasks collected.")


if __name__ == "__main__":
    main()