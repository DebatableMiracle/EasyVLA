import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from vla_diffusion import VlaDiffusion
from utils.tokenizer import tokenize_instruction

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR       = "checkpoints"
EPOCHS         = 100
BATCH_SIZE     = 64
LR             = 1e-4
STATE_DIM      = 39
ACTION_DIM     = 4
D_MODEL        = 256
ACTION_HORIZON = 8
INSTRUCTION    = "reach the target"
VAL_SPLIT      = 0.1
WANDB_PROJECT  = "vla-from-scratch"        # change to your project name
OBS_HORIZON    = 2


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


from tqdm import tqdm

def run_epoch(model, loader, text_ids, text_mask, optimizer=None):
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
            B     = img.size(0)

            t_ids  = text_ids.repeat(B, 1).to(DEVICE)
            t_mask = text_mask.repeat(B, 1).to(DEVICE)

            loss = model.loss(img, t_ids, t_mask, state, act)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")  # shows current batch loss too

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
            "instruction":    INSTRUCTION,
        }
    )

    print("Loading data...")
    train_loader, val_loader = make_loaders(VAL_SPLIT, BATCH_SIZE)


    model = VlaDiffusion(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        action_horizon=ACTION_HORIZON,
        obs_horizon=OBS_HORIZON,
        ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    text_ids, text_mask = tokenize_instruction(INSTRUCTION)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, text_ids, text_mask, optimizer)
        val_loss   = run_epoch(model, val_loader,   text_ids, text_mask)
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
            ckpt = {
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_loss,
                "config": {
                    "state_dim":      STATE_DIM,
                    "action_dim":     ACTION_DIM,
                    "d_model":        D_MODEL,
                    "action_horizon": ACTION_HORIZON,
                }
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, "best.pt"))
            wandb.log({"best_val_loss": val_loss})
            print(f"  ✓ saved best checkpoint (val {val_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()