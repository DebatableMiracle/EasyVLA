#this one is kinda claude honestly, the script is good to go so you can try it out
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from vla_diffusion import VlaDiffusion
from utils.tokenizer import tokenize_instruction

# ── config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH   = "data/reach_v3_demos.npz"
SAVE_DIR    = "checkpoints"
EPOCHS      = 20
BATCH_SIZE  = 64
LR          = 1e-4
STATE_DIM   = 39
ACTION_DIM  = 4
D_MODEL     = 256
INSTRUCTION = "reach the target"
VAL_SPLIT   = 0.1
# ──────────────────────────────────────────────────────────────────────────────


class DemoDataset(Dataset):
    def __init__(self, images, states, actions):
        self.images  = images
        self.states  = states
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.images[idx], self.states[idx], self.actions[idx]


def load_data(path):
    data    = np.load(path)
    images  = torch.tensor(data["images"]).permute(0, 3, 1, 2).float() / 255.0
    states  = torch.tensor(data["states"]).float()
    actions = torch.tensor(data["actions"]).float()
    return images, states, actions


def make_loaders(images, states, actions, val_split, batch_size):
    N       = len(actions)
    n_val   = int(N * val_split)
    n_train = N - n_val

    # reproducible split — no random shuffle so episodes stay contiguous
    train_ds = DemoDataset(images[:n_train], states[:n_train], actions[:n_train])
    val_ds   = DemoDataset(images[n_train:], states[n_train:], actions[n_train:])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def run_epoch(model, loader, text_ids, text_mask, optimizer=None):
    """Shared logic for train and val. Pass optimizer=None for eval."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for img, state, act in loader:
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

    return total_loss / len(loader)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading data...")
    images, states, actions = load_data(DATA_PATH)
    train_loader, val_loader = make_loaders(
        images, states, actions, VAL_SPLIT, BATCH_SIZE
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = VlaDiffusion(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    text_ids, text_mask = tokenize_instruction(INSTRUCTION)  
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, text_ids, text_mask, optimizer)
        val_loss   = run_epoch(model, val_loader,   text_ids, text_mask)
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(ckpt, os.path.join(SAVE_DIR, "best.pt"))
            print(f"  ✓ saved best checkpoint (val {val_loss:.4f})")

    # always save final
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final.pt"))
    print("Done.")


if __name__ == "__main__":
    main()