# VLA from Scratch

A Vision-Language-Action model built from scratch for robot manipulation, trained on MetaWorld environments. This repo is a fully modular, research-friendly implementation — every component (vision encoder, language encoder, fusion, action head) is swappable. Built alongside [this article](#) as a practical guide to understanding modern VLA architectures.

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![MetaWorld](https://img.shields.io/badge/env-MetaWorld-green)
![WandB](https://img.shields.io/badge/logging-wandb-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Results

Trained on `reach-v3` with 960 expert demonstrations:

| Metric | Value |
|---|---|
| Val loss (epoch 38) | **0.064** |
| Success rate (50 rollouts) | **56%** |
| Avg steps on success | **~30 steps** |
| Expert avg steps | ~50 steps |

Successful episodes are **faster than the expert policy** — the model learned efficient reach trajectories rather than random walks.

> MT-10 multi-task training coming soon.

---

## Architecture

```
Image (224×224×6)  ──► Vision Encoder (ResNet-18)  ──► vision tokens (B, 49, 256)
                                                               │
Instruction text   ──► Text Encoder (DistilBERT)   ──► text tokens   (B, L, 256)
                                                               │
State (39-dim)     ──► State Encoder (MLP)         ──► state token  (B, 1, 256)
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │  Fusion Transformer  │
                                                    │  (Cross-Attention)   │
                                                    │  state queries        │
                                                    │  vision+text context  │
                                                    └──────────┬──────────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │   Diffusion Head     │
                                                    │   DDPM, T=64         │
                                                    │   action chunk H=8   │
                                                    └──────────┬──────────┘
                                                               │
                                              Actions (B, 8, 4) → execute open-loop
```

### Key design choices

**Observation history (`obs_horizon=2`)** — stacks the current and previous frame along the channel dimension (6 channels total), giving the model implicit velocity information. The model can infer whether the arm is moving toward or away from the target.

**Action chunking (`action_horizon=8`)** — predicts 8 future actions at once and executes them open-loop before replanning. Borrowed from the Diffusion Policy paper. Prevents error compounding from single-step reactive control.

**Cross-attention fusion** — state tokens query the visual+text context as keys/values rather than everything being mixed through a CLS token. Allows the model to ask spatially-specific questions of the visual context.

**DDPM diffusion head** — standard DDPM with T=64 denoising steps over a flattened action chunk. SiLU activations, sinusoidal time embeddings, hidden_dim=512.

---

## Modular Design

Every component is designed to be swappable. The current implementation uses one stack — more are coming:

### Vision Encoders
| Encoder | Status | Style |
|---|---|---|
| ResNet-18 | ✅ current | ACT / Diffusion Policy |
| EfficientNet-B0 | 🔜 soon | faster alternative |
| DINOv2 | 🔜 soon | OpenVLA-style, spatial features |
| SigLIP | 🔜 soon | π0 / SmolVLA-style |
| CLIP ViT | 🔜 soon | strong vision-language baseline |

### Language Encoders
| Encoder | Status | Style |
|---|---|---|
| DistilBERT | ✅ current | lightweight transformer |
| T5-base | 🔜 soon | Octo-style |
| CLIP text | 🔜 soon | pairs with CLIP vision |
| SmolLM2 | 🔜 soon | SmolVLA-style, 135M |

### Fusion
| Type | Status | Style |
|---|---|---|
| Cross-attention transformer | ✅ current | SmolVLA-style |
| FiLM | 🔜 soon | lightweight language-gated vision |
| Self-attention (all tokens) | 🔜 soon | Octo-style |

### Action Heads
| Head | Status | Style |
|---|---|---|
| DDPM Diffusion | ✅ current | Diffusion Policy / Octo |
| Flow Matching | 🔜 soon | π0 / GR00T-style |
| ACT (CVAE) | 🔜 soon | ACT-style |
| MLP baseline | 🔜 soon | simple baseline |

---

## Project Structure

```
vla_from_scratch/
├── action_head/
│   └── diffusion_head.py       # DDPM diffusion over action chunks
├── encoders/
│   ├── vision_encoder.py       # ResNet-18 with obs stacking
│   ├── text_encoder.py         # DistilBERT with projection
│   └── state_encoder.py        # MLP state tokenizer
├── envs/
│   └── metaworld_env.py        # MetaWorld wrapper with frame buffer
├── data/
│   └── collect_data.py         # expert demo collection, chunked saving
├── utils/
│   ├── tokenizer.py            # DistilBERT tokenizer wrapper
│   └── push_to_hf.py           # push checkpoints to HuggingFace Hub
├── fusion.py                   # cross-attention fusion transformer
├── vla_diffusion.py            # main VlaModel class
├── train.py                    # training loop with wandb
└── rollout.py                  # evaluation + rendering
```

---

## Installation

```bash
git clone https://github.com/your-username/vla_from_scratch
cd vla_from_scratch

conda create -n easyvla python=3.10
conda activate easyvla

pip install torch torchvision
pip install metaworld gymnasium
pip install transformers
pip install wandb tqdm
pip install huggingface_hub
pip install opencv-python
```

---

## Usage

### 1. Collect demonstrations

```bash
python -m data.collect_data
```

Saves chunked `.npy` files to `data/` — collects 100 episodes at a time to avoid RAM overflow, appends to disk incrementally.

Key config in `collect_data.py`:
```python
TASK           = "reach-v3"
EPISODES       = 1000
IMG_SIZE       = 224
ACTION_HORIZON = 8
OBS_HORIZON    = 2
CHUNK_SIZE     = 100
```

### 2. Train

```bash
python train.py
```

Logs to wandb automatically. Saves best checkpoint by val loss to `checkpoints/best.pt`.

Key config in `train.py`:
```python
EPOCHS         = 100
BATCH_SIZE     = 64
LR             = 1e-4
D_MODEL        = 256
ACTION_HORIZON = 8
OBS_HORIZON    = 2
```

### 3. Evaluate

```bash
python rollout.py
```

Renders live with `render_mode="human"`. Reports per-episode success/fail and overall success rate.

### 4. Push to HuggingFace

```bash
export HF_TOKEN=your_token
python utils/push_to_hf.py
```

---

## Training Details

| Component | Choice | Params |
|---|---|---|
| Vision encoder | ResNet-18 (layer3+4 unfrozen) | ~12M trainable |
| Text encoder | DistilBERT (frozen + projection) | ~0.2M trainable |
| State encoder | 3-layer MLP | ~0.15M trainable |
| Fusion | Cross-attention transformer, 4 layers | ~3.5M trainable |
| Diffusion head | DDPM, T=64, hidden=512 | ~2.5M trainable |
| **Total trainable** | | **~18M / 80M** |

Optimizer: AdamW, lr=1e-4, cosine decay, weight_decay=1e-4  
Gradient clipping: max_norm=1.0  
Dataset: 960 episodes × ~50 steps → 48k `(obs, action_chunk)` pairs

---

## Roadmap

- [x] Single task (reach-v3)
- [x] Action chunking
- [x] Observation history
- [x] Cross-attention fusion
- [x] WandB logging
- [x] HuggingFace push
- [ ] Modular encoder/fusion/head registry
- [ ] MT-10 multi-task training
- [ ] DINOv2 vision encoder
- [ ] Flow matching action head
- [ ] 84×84 resolution for MT-10 storage efficiency
- [ ] Per-task success rate evaluation
- [ ] Pre-trained model weights on HuggingFace

---

## References

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Chi et al., action chunking + DDPM over trajectories
- [ACT](https://tonyzhaozh.github.io/aloha/) — Zhao et al., action chunking with transformers
- [Octo](https://octo-models.github.io/) — cross-attention fusion, diffusion head
- [SmolVLA](https://huggingface.co/blog/smolvla) — lightweight VLA, similar fusion design
- [OpenVLA](https://openvla.github.io/) — LLM-based VLA backbone
- [π0](https://www.physicalintelligence.company/blog/pi0) — flow matching over joint tokens
- [MetaWorld](https://meta-world.github.io/) — benchmark environments

---

## License

MIT
