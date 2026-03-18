import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark       = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

from vla_diffusion import VlaDiffusion
from encoders.registry import build_text_encoder
from utils.tokenizer import tokenize_instruction

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR       = "checkpoints"
EPOCHS         = 10
BATCH_SIZE     = 64
LR             = 1e-4
STATE_DIM      = 39
ACTION_DIM     = 4
D_MODEL        = 256
ACTION_HORIZON = 8
OBS_HORIZON    = 2
VISION_ENCODER = "resnet18"    # "resnet18" | "efficientnet" | "mobilenet" | "dinov2"
TEXT_ENCODER   = "distilbert"  # "distilbert" | "smollm" | "bert_tiny"
INSTRUCTION    = "reach the target"
VAL_SPLIT      = 0.1
WANDB_PROJECT  = "vla-from-scratch"


class DemoDataset(Dataset):
    def __init__(self, indices):
        self.images  = np.load("data/images.npy",  mmap_mode="r")
        self.states  = np.load("data/states.npy",  mmap_mode="r")
        self.actions = np.load("data/actions.npy", mmap_mode="r")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i      = self.indices[idx]
        image  = torch.tensor(self.images[i].copy()).permute(2, 0, 1).float() / 255.0
        state  = torch.tensor(self.states[i].copy()).float()
        action = torch.tensor(self.actions[i].copy()).float()
        return image, state, action


def make_loaders(val_split, batch_size):
    N       = np.load("data/images.npy", mmap_mode="r").shape[0]
    n_val   = int(N * val_split)
    n_train = N - n_val
    indices = np.arange(N)
    print(f"Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(DemoDataset(indices[:n_train]), batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(DemoDataset(indices[n_train:]), batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def precompute_text_tokens(instruction, text_encoder_name, d_model):
    """Run text encoder once on CPU, delete it, return token tensor."""
    print(f"Precomputing text tokens with {text_encoder_name}...")
    encoder = build_text_encoder(text_encoder_name, d_model)   # CPU
    ids, mask = tokenize_instruction(instruction)
    with torch.no_grad():
        tokens = encoder(ids, mask)                             # (1, L, d_model)
    del encoder                                                 # free memory immediately
    print(f"Text tokens shape: {tuple(tokens.shape)} — encoder freed from memory")
    return tokens.detach()                                      # (1, L, d_model) on CPU


def run_epoch(model, loader, text_tokens, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc="train" if is_train else "val  ", leave=False)
        for img, state, act in pbar:
            img   = img.to(DEVICE)
            state = state.to(DEVICE)
            act   = act.to(DEVICE)

            loss = model.loss(img, text_tokens, state, act)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        config={
            "epochs":         EPOCHS,
            "batch_size":     BATCH_SIZE,
            "lr":             LR,
            "state_dim":      STATE_DIM,
            "action_dim":     ACTION_DIM,
            "d_model":        D_MODEL,
            "action_horizon": ACTION_HORIZON,
            "obs_horizon":    OBS_HORIZON,
            "vision_encoder": VISION_ENCODER,
            "text_encoder":   TEXT_ENCODER,
            "instruction":    INSTRUCTION,
        }
    )

    # precompute text tokens — encoder never touches GPU
    text_tokens = precompute_text_tokens(INSTRUCTION, TEXT_ENCODER, D_MODEL)

    print("Loading data...")
    train_loader, val_loader = make_loaders(VAL_SPLIT, BATCH_SIZE)

    # text encoder not loaded into model — vision/state/fusion/diffusion only
    model = VlaDiffusion(
        state_dim      = STATE_DIM,
        action_dim     = ACTION_DIM,
        d_model        = D_MODEL,
        action_horizon = ACTION_HORIZON,
        obs_horizon    = OBS_HORIZON,
        vision_encoder = VISION_ENCODER,
        text_encoder   = TEXT_ENCODER,   # still needed to build projection layer
    ).to(DEVICE)

    # free the text encoder weights from the model too — keep only projection
    del model.text_encoder
    model.text_encoder = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, text_tokens, optimizer)
        val_loss   = run_epoch(model, val_loader,   text_tokens)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.2e}")

        wandb.log({
            "train/loss": train_loss,
            "val/loss":   val_loss,
            "lr":         lr,
            "epoch":      epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  val_loss,
                "text_tokens": text_tokens,   # save tokens so rollout doesn't need encoder
                "config": {
                    "state_dim":      STATE_DIM,
                    "action_dim":     ACTION_DIM,
                    "d_model":        D_MODEL,
                    "action_horizon": ACTION_HORIZON,
                    "obs_horizon":    OBS_HORIZON,
                    "vision_encoder": VISION_ENCODER,
                    "text_encoder":   TEXT_ENCODER,
                }
            }, os.path.join(SAVE_DIR, "best.pt"))
            wandb.log({"best_val_loss": val_loss})
            print(f"  ✓ saved best checkpoint (val {val_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()