# EasyVLA

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

## Ongoing Experiments

Currently training the model across 5 MetaWorld tasks using DINOv2 as the vision encoder, Distilbert as the text encoder, and MLP as the state encoder to improve representation quality and generalization. I've moved from 224*224 to 84*84 images to reduce training time and memory usage. At the same time, since the states control dx,dy,dz to reduce jitter, I beleived acceleration component is also important hence I moved from OBS_HORIZON=2 to OBS_HORIZON=3. 

These experiments aim to evaluate:
- Cross-task generalization
- Sample efficiency across multiple tasks
- Impact of stronger visual representations on policy learning

Results will be added upon completion of training.
---

## Architecture

```
Image (84×84×6)    ──► Vision Encoder (ResNet-18)  ──► vision tokens (B, N, 128)
                                                               │
Instruction text   ──► Text Encoder (precomputed)  ──► text tokens   (1, L, 128)
                                                               │
State (39-dim)     ──► State Encoder (MLP)         ──► state token  (B, 1, 128)
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

**Precomputed text tokens** — the text encoder runs once on CPU at startup and is deleted from memory. Since the instruction never changes within a task, there's no reason to keep 66M DistilBERT params on GPU. Tokens are saved inside the checkpoint so rollout doesn't need the encoder at all.

---

## Modular Design

Every component is swappable via a string in `train.py`. No code changes needed — just change the config:

```python
VISION_ENCODER = "efficientnet"   # swap vision encoder
TEXT_ENCODER   = "bert_tiny"      # swap text encoder
```

### Vision Encoders
| Encoder | Status | Params | Style |
|---|---|---|---|
| ResNet-18 | ✅ available | 11M | ACT / Diffusion Policy |
| EfficientNet-B0 | ✅ available | 5.3M | faster alternative |
| MobileNetV3-Small | ✅ available | 2.5M | lightweight, edge-friendly |
| DINOv2-Small | ✅ available | 22M | OpenVLA-style, spatial features |
| CLIP ViT-B/32 | 🔜 soon | 87M | vision-language aligned |
| SigLIP | 🔜 soon | 400M | π0 / SmolVLA-style |
| R3M | 🔜 soon | 26M | manipulation-specific pretraining |

### Language Encoders
| Encoder | Status | Params | Style |
|---|---|---|---|
| DistilBERT | ✅ available | 66M frozen | lightweight transformer |
| SmolLM2-135M | ✅ available | 135M frozen | SmolVLA-style |
| BERT-Tiny | ✅ available | 4.4M frozen | fastest option |
| CLIP text | 🔜 soon | 63M | pairs with CLIP vision |
| T5-base | 🔜 soon | 220M | Octo-style |

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
EasyVLA/
├── action_head/
│   └── diffusion_head.py          # DDPM diffusion over action chunks
├── encoders/
│   ├── registry.py                # encoder registry — add new encoders here
│   ├── vision/
│   │   ├── base.py                # BaseVisionEncoder ABC
│   │   ├── resnet.py              # ResNet-18
│   │   ├── efficientnet.py        # EfficientNet-B0
│   │   ├── mobilenet.py           # MobileNetV3-Small
│   │   └── dinov2.py              # DINOv2-Small
│   ├── text/
│   │   ├── base.py                # BaseTextEncoder ABC
│   │   ├── distilbert.py          # DistilBERT
│   │   ├── smollm.py              # SmolLM2-135M
│   │   └── bert_tiny.py           # BERT-Tiny
│   └── state/
│       ├── base.py                # BaseStateEncoder ABC
│       └── mlp.py                 # 3-layer MLP
├── envs/
│   └── metaworld_env.py           # MetaWorld wrapper with frame buffer
├── data/
│   └── collect_data.py            # expert demo collection, chunked saving
├── utils/
│   ├── tokenizer.py               # tokenizer wrapper
│   └── push_to_hf.py              # push checkpoints to HuggingFace Hub
├── fusion.py                      # cross-attention fusion transformer
├── vla_diffusion.py               # main VlaModel class
├── train.py                       # training loop with wandb
└── rollout.py                     # evaluation + rendering
```

---

## Installation

```bash
git clone https://github.com/DebatableMiracle/EasyVLA.git
cd EasyVLA

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
IMG_SIZE       = 84
ACTION_HORIZON = 8
OBS_HORIZON    = 2
CHUNK_SIZE     = 100
```

### 2. Train

```bash
python train.py
```

Logs to wandb automatically. Saves best checkpoint by val loss to `checkpoints/best.pt`. Text encoder runs once on CPU at startup then is freed — only vision, fusion, and diffusion head live on GPU during training.

Key config in `train.py`:
```python
EPOCHS         = 100
BATCH_SIZE     = 64
LR             = 1e-4
D_MODEL        = 128
ACTION_HORIZON = 8
OBS_HORIZON    = 2
VISION_ENCODER = "resnet18"    # resnet18 | efficientnet | mobilenet | dinov2
TEXT_ENCODER   = "distilbert"  # distilbert | smollm | bert_tiny
```

### 3. Evaluate

```bash
python rollout.py
```

Renders live with `render_mode="human"`. Loads text tokens directly from checkpoint — no text encoder needed at rollout time.

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
| Text encoder | precomputed on CPU, freed after | 0 on GPU |
| State encoder | 3-layer MLP | ~0.05M trainable |
| Fusion | Cross-attention transformer, 4 layers | ~1M trainable |
| Diffusion head | DDPM, T=64, hidden=512 | ~2.5M trainable |
| **Total trainable** | | **~15M / 80M** |

Optimizer: AdamW, lr=1e-4, cosine decay, weight_decay=1e-4  
Gradient clipping: max_norm=1.0  
Dataset: 960 episodes × ~50 steps → 48k `(obs, action_chunk)` pairs

---

## Patch Notes

### v0.3 — March 18, 2026
- **Modular encoder registry** — all encoders swappable via string config, no code changes needed
- **New vision encoders** — EfficientNet-B0, MobileNetV3-Small, DINOv2-Small
- **New text encoders** — SmolLM2-135M, BERT-Tiny
- **Base classes** — `BaseVisionEncoder`, `BaseTextEncoder`, `BaseStateEncoder` ABCs for clean extension
- **Precomputed text tokens** — text encoder runs once on CPU at startup, freed from GPU entirely. Tokens saved inside checkpoint so rollout needs no encoder
- **Dynamic `n_tokens`** — vision encoders auto-compute token count from input resolution, no manual updates when changing image size
- **cuDNN + TF32 flags** — `cudnn.benchmark=True` and TF32 enabled for Ampere GPUs
- **`d_model=128`** — halved model width for 4GB GPU compatibility and MT10 scalability
- **`IMG_SIZE=84`** — moving to 84×84 for storage efficiency at MT10 scale

### v0.2 — March 2026
- **Action chunking** — predict `action_horizon=8` future actions, execute open-loop before replanning
- **Observation history** — `obs_horizon=2` stacks current + previous frame, gives model velocity signal
- **Cross-attention fusion** — replaced CLS token self-attention with state-queries-visual-context design
- **Bigger denoiser** — `hidden_dim=512`, `time_emb_dim=64`, `T=64` diffusion steps
- **Deeper fusion** — 4 transformer layers, 8 attention heads
- **Wider state encoder** — 3-layer MLP, `hidden=256`
- **Incremental data collection** — chunked saving every 100 episodes to avoid RAM crash
- **mmap dataset** — `.npy` files with `mmap_mode="r"`, full dataset never loads into RAM
- **WandB logging** — full training metrics, encoder config logged per run
- **Encoder config in checkpoint** — `best.pt` stores full config so rollout always matches training

### v0.1 — March 2026
- Initial implementation — single task reach-v3
- ResNet-18 vision encoder, DistilBERT text encoder, MLP state encoder
- Self-attention fusion with CLS token
- DDPM diffusion head, T=16, single action prediction
- Basic training loop and rollout script

---

## Roadmap

- [x] Single task (reach-v3)
- [x] Action chunking
- [x] Observation history
- [x] Cross-attention fusion
- [x] WandB logging
- [x] HuggingFace push
- [x] Modular encoder registry
- [x] EfficientNet, MobileNet, DINOv2 vision encoders
- [x] SmolLM2, BERT-Tiny text encoders
- [x] Precomputed text tokens — zero text encoder memory on GPU
- [ ] MT-10 multi-task training
- [ ] 84×84 dataset collection + training
- [ ] Flow matching action head
- [ ] CLIP vision + text encoder pair
- [ ] SigLIP vision encoder
- [ ] R3M manipulation-specific encoder
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