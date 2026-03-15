import torch
import numpy as np
from envs.metaworld_env import MetaWorldEnv
from vla_diffusion import VlaDiffusion
from utils.tokenizer import tokenize_instruction

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH     = "checkpoints/best.pt"
STATE_DIM      = 39
ACTION_DIM     = 4
D_MODEL        = 256
ACTION_HORIZON = 16           # NEW: must match training
INSTRUCTION    = "reach the target"
MAX_STEPS      = 200
EPISODES       = 10


def load_model():
    model = VlaDiffusion(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        action_horizon=ACTION_HORIZON,
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_episode(env, model, text_ids, text_mask):
    obs     = env.reset()
    done    = False
    step    = 0
    success = False

    while not done and step < MAX_STEPS:
        image = torch.tensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state = torch.tensor(obs["state"]).unsqueeze(0).float()

        with torch.no_grad():
            action_chunk = model.act(image.to(DEVICE), text_ids, text_mask, state.to(DEVICE))
            # action_chunk: (1, H, action_dim)

        # execute all H actions open-loop before re-planning
        for h in range(ACTION_HORIZON):
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
    model = load_model()
    print(f"Loaded model from {MODEL_PATH}")

    text_ids, text_mask = tokenize_instruction(INSTRUCTION)
    text_ids  = text_ids.to(DEVICE)
    text_mask = text_mask.to(DEVICE)

    env       = MetaWorldEnv("reach-v3", render_mode="human")
    successes = 0

    for ep in range(EPISODES):
        success, steps = run_episode(env, model, text_ids, text_mask)
        successes += int(success)
        print(f"Episode {ep+1:02d} | {'SUCCESS' if success else 'FAIL'} | steps: {steps}")

    print(f"\nSuccess rate: {successes}/{EPISODES} ({100*successes/EPISODES:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()