import torch
import numpy as np
from envs.metaworld_env import MetaWorldEnv
from vla_diffusion import VlaDiffusion
from utils.task_config import TASK_INSTRUCTIONS

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/best.pt"
MAX_STEPS  = 200
EPISODES   = 10

# ── choose task to evaluate ───────────────────────────────────────────────────
EVAL_TASK = "reach-v3"
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
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

    model.load_state_dict(ckpt["model"])
    del model.text_encoder
    model.text_encoder = None
    model.eval()

    # find task_id for eval task
    tasks      = ckpt["tasks"]
    token_dict = ckpt["token_dict"]

    if EVAL_TASK not in tasks:
        raise ValueError(f"{EVAL_TASK} not in checkpoint tasks: {tasks}")

    task_id     = tasks.index(EVAL_TASK)
    text_tokens = token_dict[task_id].to(DEVICE)  # (1, L, d_model)

    return model, text_tokens, cfg["action_horizon"], cfg["obs_horizon"]


def run_episode(env, model, text_tokens, action_horizon):
    obs     = env.reset()
    done    = False
    step    = 0
    success = False

    while not done and step < MAX_STEPS:
        image = torch.tensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state = torch.tensor(obs["state"]).unsqueeze(0).float()

        with torch.no_grad():
            action_chunk = model.act(
                image.to(DEVICE),
                text_tokens,
                state.to(DEVICE)
            )

        for h in range(action_horizon):
            if done or step >= MAX_STEPS:
                break
            action = action_chunk[0, h].cpu().numpy()
            obs, _, done, info = env.step(action)
            step += 1
            if int(info.get("success", 0)) == 1:
                success = True
                done    = True

    return success, step


def main():
    model, text_tokens, action_horizon, obs_horizon = load_model()
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Evaluating: {EVAL_TASK} — '{TASK_INSTRUCTIONS[EVAL_TASK]}'")

    env       = MetaWorldEnv(EVAL_TASK, render_mode="human", obs_horizon=obs_horizon)
    successes = 0

    for ep in range(EPISODES):
        success, steps = run_episode(env, model, text_tokens, action_horizon)
        successes += int(success)
        print(f"Episode {ep+1:02d} | {'SUCCESS' if success else 'FAIL'} | steps: {steps}")

    print(f"\nSuccess rate: {successes}/{EPISODES} ({100*successes/EPISODES:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()