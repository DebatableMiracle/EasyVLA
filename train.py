import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

from vla_diffusion import VlaDiffusion
from encoders.registry import build_text_encoder
from utils.tokenizer import tokenize_instruction
from utils.task_config import TASK_INSTRUCTIONS

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR       = "checkpoints"
EPOCHS         = 100
BATCH_SIZE     = 96
LR             = 1e-4
STATE_DIM      = 39
ACTION_DIM     = 4
D_MODEL        = 256
ACTION_HORIZON = 8
OBS_HORIZON    = 3
VISION_ENCODER = "dinov2"
TEXT_ENCODER   = "distilbert"
VAL_SPLIT      = 0.1
WANDB_PROJECT  = "vla-from-scratch"
DATA_ROOT      = "data"

# ── choose tasks to train on ──────────────────────────────────────────────────
TASKS = [
    "reach-v3",           # easy   — baseline, you know what good looks like
    "drawer-close-v3",    # easy   — short motion, high success rate
    "button-press-topdown-v3",  # easy/medium — requires precision, not just reaching
    "door-open-v3",       # medium — requires contact + sustained force
    "push-v3",            # medium — requires object interaction
]
# ─────────────────────────────────────────────────────────────────────────────


class TaskDataset(Dataset):
    def __init__(self, task_name, task_id, indices):
        task_dir         = os.path.join(DATA_ROOT, task_name)
        self.images      = np.load(os.path.join(task_dir, "images.npy"),  mmap_mode="r")
        self.states      = np.load(os.path.join(task_dir, "states.npy"),  mmap_mode="r")
        self.actions     = np.load(os.path.join(task_dir, "actions.npy"), mmap_mode="r")
        self.task_id     = task_id
        self.indices     = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i      = self.indices[idx]
        image  = torch.tensor(self.images[i].copy()).permute(2, 0, 1).float() / 255.0
        state  = torch.tensor(self.states[i].copy()).float()
        action = torch.tensor(self.actions[i].copy()).float()
        return image, state, action, self.task_id


def make_loaders(tasks, val_split, batch_size):
    train_datasets = []
    val_datasets   = []

    for task_id, task_name in enumerate(tasks):
        N       = np.load(os.path.join(DATA_ROOT, task_name, "images.npy"),
                          mmap_mode="r").shape[0]
        n_val   = int(N * val_split)
        n_train = N - n_val
        indices = np.arange(N)

        train_datasets.append(TaskDataset(task_name, task_id, indices[:n_train]))
        val_datasets.append(TaskDataset(task_name, task_id, indices[n_train:]))

        print(f"  {task_name:<35} train: {n_train:>6} | val: {n_val:>5}")

    # ConcatDataset merges all tasks — joint training
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def precompute_text_tokens(tasks, text_encoder_name, d_model):
    print(f"Precomputing text tokens for {len(tasks)} tasks...")
    encoder = build_text_encoder(text_encoder_name, d_model)
    token_dict = {}

    for task_id, task_name in enumerate(tasks):
        instruction = TASK_INSTRUCTIONS[task_name]
        ids, mask   = tokenize_instruction(instruction)
        with torch.no_grad():
            tokens = encoder(ids, mask)          # (1, L, d_model)
        token_dict[task_id] = tokens.detach()
        print(f"  [{task_id}] {task_name}: '{instruction}'")

    del encoder
    print("Text encoder freed from memory.")
    return token_dict  # {task_id: (1, L, d_model)}


def run_epoch(model, loader, token_dict, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(loader, desc="train" if is_train else "val  ", leave=False)
        for img, state, act, task_ids in pbar:
            img   = img.to(DEVICE)
            state = state.to(DEVICE)
            act   = act.to(DEVICE)
            B     = img.size(0)

            # build per-sample text tokens from task_ids
            text_tokens = torch.cat([
                token_dict[t.item()].expand(1, -1, -1)
                for t in task_ids
            ], dim=0).to(DEVICE)   # (B, L, d_model)

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
            "tasks":          TASKS,
        }
    )

    token_dict = precompute_text_tokens(TASKS, TEXT_ENCODER, D_MODEL)

    print("\nLoading data...")
    train_loader, val_loader = make_loaders(TASKS, VAL_SPLIT, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VlaDiffusion(
        state_dim      = STATE_DIM,
        action_dim     = ACTION_DIM,
        d_model        = D_MODEL,
        action_horizon = ACTION_HORIZON,
        obs_horizon    = OBS_HORIZON,
        vision_encoder = VISION_ENCODER,
        text_encoder   = TEXT_ENCODER,
    ).to(DEVICE)

    del model.text_encoder
    model.text_encoder = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, token_dict, optimizer)
        val_loss   = run_epoch(model, val_loader,   token_dict)
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
                "epoch":       epoch,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    val_loss,
                "token_dict":  token_dict,
                "tasks":       TASKS,
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