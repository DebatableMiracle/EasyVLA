import torch
import numpy as np
from envs.metaworld_env import MetaWorldEnv
from vla_diffusion import VlaDiffusion
from utils.tokenizer import tokenize_instruction

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH  = "checkpoints/best.pt"
STATE_DIM   = 39
ACTION_DIM  = 4
D_MODEL     = 256
INSTRUCTION = "reach the target"
MAX_STEPS   = 200
EPISODES    = 10


def load_model():
    model = VlaDiffusion(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
    ).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])   # checkpoint is a dict, not raw state_dict
    model.eval()
    return model


def run_episode(env, model, text_ids, text_mask):
    obs = env.reset()
    done = False
    step = 0
    success = False

    while not done and step < MAX_STEPS:
        image = torch.tensor(obs["image"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        state = torch.tensor(obs["state"]).unsqueeze(0).float()

        image = image.to(DEVICE)
        state = state.to(DEVICE)

        with torch.no_grad():
            action = model.act(image, text_ids, text_mask, state)

        action = action.cpu().numpy()[0]
        obs, _, done, info = env.step(action)

        if int(info.get("success", 0)) == 1:
            success = True
            done = True

        step += 1

    return success, step


def main():
    model = load_model()
    print(f"Loaded model from {MODEL_PATH}")

    text_ids, text_mask = tokenize_instruction(INSTRUCTION)
    text_ids  = text_ids.to(DEVICE)
    text_mask = text_mask.to(DEVICE)

    env = MetaWorldEnv("reach-v3")

    successes = 0
    for ep in range(EPISODES):
        success, steps = run_episode(env, model, text_ids, text_mask)
        successes += int(success)
        print(f"Episode {ep+1:02d} | {'SUCCESS' if success else 'FAIL'} | steps: {steps}")

    print(f"\nSuccess rate: {successes}/{EPISODES} ({100*successes/EPISODES:.0f}%)")
    env.close()


if __name__ == "__main__":
    main()