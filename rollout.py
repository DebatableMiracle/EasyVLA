import torch
import numpy as np
import cv2
import os
import argparse
from envs.metaworld_env import MetaWorldEnv
from vla_diffusion import VlaDiffusion
from utils.task_config import TASK_INSTRUCTIONS

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best.pt"
MAX_STEPS  = 200
EPISODES   = 10

VIDEO_DIR  = "videos"
VIDEO_FPS  = 20

EVAL_TASKS = [
    "reach-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "door-open-v3",
    "push-v3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="EasyVLA Rollout")
    parser.add_argument("--tasks",    nargs="+", default=EVAL_TASKS,
                        help="tasks to evaluate")
    parser.add_argument("--episodes", type=int,  default=EPISODES,
                        help="episodes per task")
    parser.add_argument("--mode",     type=str,  default="render",
                        choices=["render", "save_video", "headless"],
                        help="render=live window | save_video=mp4 | headless=no display")
    parser.add_argument("--model",    type=str,  default=MODEL_PATH,
                        help="path to checkpoint")
    parser.add_argument("--max_steps",type=int,  default=MAX_STEPS)
    return parser.parse_args()


def load_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    cfg  = ckpt["config"]

    model = VlaDiffusion(
        state_dim      = cfg["state_dim"],
        action_dim     = cfg["action_dim"],
        d_model        = cfg["d_model"],
        action_horizon = cfg["action_horizon"],
        obs_horizon    = cfg["obs_horizon"],
        vision_encoder = cfg["vision_encoder"],
        text_encoder   = cfg["text_encoder"],
    ).to(DEVICE)

    del model.text_encoder
    model.text_encoder = None
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, ckpt["token_dict"], ckpt["tasks"], cfg["action_horizon"], cfg["obs_horizon"]


def run_episode(env, model, text_tokens, action_horizon, max_steps, save_video=False):
    obs     = env.reset()
    done    = False
    step    = 0
    success = False
    frames  = []

    while not done and step < max_steps:
        image = torch.tensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state = torch.tensor(obs["state"]).unsqueeze(0).float()

        if save_video:
            frame = env.env.render()
            if frame is not None:
                frames.append(frame)

        with torch.no_grad():
            action_chunk = model.act(
                image.to(DEVICE),
                text_tokens,
                state.to(DEVICE)
            )

        for h in range(action_horizon):
            if done or step >= max_steps:
                break
            action = action_chunk[0, h].cpu().numpy()
            obs, _, done, info = env.step(action)
            step += 1
            if int(info.get("success", 0)) == 1:
                success = True
                done    = True

    return success, step, frames


def save_video_file(frames, path):
    if not frames:
        return
    h, w   = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    args = parse_args()

    model, token_dict, ckpt_tasks, action_horizon, obs_horizon = load_model(args.model)
    print(f"Loaded model from {args.model}")
    print(f"Mode: {args.mode} | Episodes: {args.episodes} | Tasks: {args.tasks}\n")

    # set render mode based on arg
    render_mode = {
        "render":     "human",
        "save_video": "rgb_array",
        "headless":   "rgb_array",
    }[args.mode]

    total_success  = 0
    total_episodes = 0
    results        = {}

    for task in args.tasks:
        if task not in ckpt_tasks:
            print(f"Skipping {task} — not in checkpoint tasks {ckpt_tasks}")
            continue

        task_id     = ckpt_tasks.index(task)
        text_tokens = token_dict[task_id].to(DEVICE)

        if args.mode == "save_video":
            task_dir = os.path.join(VIDEO_DIR, task)
            os.makedirs(task_dir, exist_ok=True)

        env = MetaWorldEnv(task, render_mode=render_mode, obs_horizon=obs_horizon)

        print(f"── {task} — '{TASK_INSTRUCTIONS[task]}' ──")
        successes = 0

        for ep in range(args.episodes):
            success, steps, frames = run_episode(
                env, model, text_tokens, action_horizon,
                max_steps  = args.max_steps,
                save_video = args.mode == "save_video",
            )
            successes += int(success)
            status = "SUCCESS" if success else "FAIL"

            if args.mode == "save_video" and frames:
                video_path = os.path.join(task_dir, f"ep{ep+1:02d}.mp4")
                save_video_file(frames, video_path)
                print(f"  Episode {ep+1:02d} | {status} | steps: {steps} | {video_path}")
            else:
                print(f"  Episode {ep+1:02d} | {status} | steps: {steps}")

        rate = 100 * successes / args.episodes
        results[task]  = rate
        total_success  += successes
        total_episodes += args.episodes
        print(f"  {task} success rate: {successes}/{args.episodes} ({rate:.0f}%)\n")
        env.close()

    # summary
    print(f"{'='*50}")
    print(f"{'Task':<35} {'Success Rate':>12}")
    print(f"{'='*50}")
    for task, rate in results.items():
        print(f"  {task:<33} {rate:>10.0f}%")
    print(f"{'='*50}")
    print(f"  {'Overall':<33} {100*total_success/total_episodes:>10.0f}%")

    if args.mode == "save_video":
        print(f"\nVideos saved to {VIDEO_DIR}/")


if __name__ == "__main__":
    main()