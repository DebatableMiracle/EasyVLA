# EasyVLA

A Vision-Language-Action model built from scratch for robot manipulation, trained on MetaWorld environments. This repo is a fully modular, research-friendly implementation — every component (vision encoder, language encoder, fusion, action head) is swappable. Built alongside [this article](#) as a practical guide to understanding modern VLA architectures.

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![MetaWorld](https://img.shields.io/badge/env-MetaWorld-green)
![WandB](https://img.shields.io/badge/logging-wandb-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Results

### Run 1 — Single Task, Baseline (reach-v3)
ResNet-18 · DistilBERT · 224×224 · obs_horizon=2 · single action prediction

| Metric | Value |
|---|---|
| Val loss (epoch 100) | **0.86** |
| Success rate (10 rollouts) | **20%** |
| Avg steps on success | ~150 steps |
| Expert avg steps | ~50 steps |

---

### Run 2 — Single Task, Improved Architecture (reach-v3)
ResNet-18 · DistilBERT · 224×224 · obs_horizon=2 · action chunking · cross-attention fusion

| Metric | Value |
|---|---|
| Val loss (epoch 38) | **0.064** |
| Success rate (50 rollouts) | **56%** |
| Avg steps on success | **~30 steps** |
| Expert avg steps | ~50 steps |

Successful episodes were **faster than the expert policy**. The main remaining issue was the camera angle — default MetaWorld `behindGripper` view caused depth ambiguity for certain target positions.

---

### Run 3 — Multi-Task MT5, DINOv2 (ongoing → recollecting)
DINOv2-Small · DistilBERT · 84×84 · obs_horizon=3 · action chunking · cross-attention fusion · 5 tasks jointly

| Task | Success Rate | Notes |
|---|---|---|
| reach-v3 | **90%** | strong |
| button-press-topdown-v3 | **100%** | strong |
| door-open-v3 | **60%** | contact task, decent |
| drawer-close-v3 | **40%** | contact task |
| push-v3 | **0%** | object interaction — camera issue |
| **Overall** | **58%** | epoch 45/100, stopped early |

> **Known issue:** entire dataset was collected using the default MetaWorld `behindGripper` camera — an ego-centric view from behind the gripper that provides poor depth perception and makes object positions ambiguous. Push-v3 fails completely because the puck and target are on the table surface which is nearly invisible from this angle. **Currently recollecting all data with `corner2` camera at 128×128 resolution.** I'm currently trying to fix this however, I have to do this with minimal dataset size and also, update to 224 or 128 res which will increase the dataset size even more.

---

## Ongoing — Run 4 (in progress)
DINOv2-Small · DistilBERT · **128×128** · obs_horizon=3 · **corner2 camera** · 5 tasks jointly

Fixing the camera angle and resolution identified as the primary bottleneck from Run 3. Expected improvement:
- Push-v3: 0% → 60%+ (table surface now visible)
- Drawer/door: 40-60% → 70%+ (better depth perception for contact)
- Overall: 58% → 75%+

Results will be added upon completion.
20 March 2026
The results haven't been better yet. Which urges me to look at the architecture more deeply. While, the architecture is essentially too weak to generalise, I still think it should be possible to arrive at some sense of generalization with ~15M trainable parameters. While you can always pinpoint that my datasets are too small and less diverse at 500-1000 episodes per task (5tasks). I think that we can still get some improvement over our current scenarios. Essentially even a simple MLP is good to solve some tasks like Reach etc. However cross attention urges the model to also use the vision tokens. I've been thinking of improving the fusion steps, add more size to outputs of the encoders. The size representation is an interesting idea for such small sized fusion. maybe too less steps to generalize over 5 different tasks honestly I train for ~50 epochs only usually however considering the loss DOES go below 5% each run, it tells me a story that model are preferring to learn trajectories rather than generalize through tasks.

My current ideas:
- Dropoutsss, drop vision tokens, state info, or even text description.
- Add moore task descriptions and cycle through them throughout training giving more genralized task solutions, however ths is about language semantics, while current issue is more around it understanding vision tokens. 
- Maybe train without state dimensions, and only use vision and text tokens. I wanna see how long it takes for that. Because there's a high chance that the model is just using the state information to solve the tasks as Camera angle changes didn't reflect much into the loss curves (WHICH SHOULD BE MY BIGGEST SIGNAL THROUGH THIS WORK). 

Anyways I'll keep updating lol
---

## Architecture

```
Image (128×128×9)  ──► Vision Encoder (DINOv2-Small) ──► vision tokens (B, N, 256)
                                                               │
Instruction text   ──► Text Encoder (precomputed)    ──► text tokens   (1, L, 256)
                                                               │
State (39-dim)     ──► State Encoder (MLP)           ──► state token  (B, 1, 256)
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

**Observation history (`obs_horizon=3`)** — stacks the last 3 frames along the channel dimension (9 channels total). Gives the model position, velocity, and acceleration signals implicitly. The model can infer whether the arm is moving toward or away from the target and whether it is decelerating for a precise approach.

**Action chunking (`action_horizon=8`)** — predicts 8 future actions at once and executes them open-loop before replanning. Borrowed from the Diffusion Policy paper. Prevents error compounding from single-step reactive control.

**Cross-attention fusion** — state tokens query the visual+text context as keys/values rather than everything being mixed through a CLS token. Allows the model to ask spatially-specific questions of the visual context.

**DDPM diffusion head** — standard DDPM with T=64 denoising steps over a flattened action chunk. SiLU activations, sinusoidal time embeddings with learned projection, hidden_dim=512, condition projection MLP.

**Precomputed text tokens** — the text encoder runs once on CPU at startup and is deleted from memory. Since the instruction never changes within a task, there's no reason to keep 66M DistilBERT params on GPU. For MT training, one token tensor is precomputed per task. Tokens are saved inside the checkpoint so rollout needs no encoder at all.

**Task-specific cameras** — different MetaWorld tasks benefit from different camera angles. The `corner2` camera provides a full workspace view with better depth cues for contact tasks. Camera is configurable per task via `TASK_CAMERAS` in `metaworld_env.py`.

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
│   └── metaworld_env.py           # MetaWorld wrapper with frame buffer + camera control
├── data/
│   └── collect_data.py            # expert demo collection, chunked saving, multi-task
├── utils/
│   ├── task_config.py             # task instructions + difficulty tiers
│   ├── tokenizer.py               # tokenizer wrapper
│   └── push_to_hf.py              # push checkpoints to HuggingFace Hub
├── fusion.py                      # cross-attention fusion transformer
├── vla_diffusion.py               # main VlaModel class
├── train.py                       # training loop with wandb, multi-task
└── rollout.py                     # evaluation + rendering + video saving
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

Saves chunked `.npy` files to `data/<task>/` — collects 100 episodes at a time to avoid RAM overflow, appends to disk incrementally. Edit `TASKS` at the top of `collect_data.py` to choose which tasks to collect.

Key config in `collect_data.py`:
```python
TASKS          = ["reach-v3", "push-v3", ...]  # choose your tasks
EPISODES       = 1000
IMG_SIZE       = 128
ACTION_HORIZON = 8
OBS_HORIZON    = 3
CHUNK_SIZE     = 100
```

### 2. Train

```bash
python train.py
```

Logs to wandb automatically. Saves best checkpoint by val loss to `checkpoints/best.pt`. Text encoders are precomputed once per task on CPU then freed — only vision, fusion, and diffusion head live on GPU during training.

Key config in `train.py`:
```python
TASKS          = ["reach-v3", "push-v3", ...]
EPOCHS         = 100
BATCH_SIZE     = 96
LR             = 1e-4
D_MODEL        = 256
ACTION_HORIZON = 8
OBS_HORIZON    = 3
VISION_ENCODER = "dinov2"      # resnet18 | efficientnet | mobilenet | dinov2
TEXT_ENCODER   = "distilbert"  # distilbert | smollm | bert_tiny
```

### 3. Evaluate

```bash
# live render
python rollout.py

# save videos to videos/ folder
python rollout.py --mode save_video

# headless, just numbers
python rollout.py --mode headless

# specific tasks, more episodes
python rollout.py --tasks reach-v3 push-v3 --episodes 20 --mode headless
```

### 4. Push to HuggingFace

```bash
export HF_TOKEN=your_token
python utils/push_to_hf.py
```

---

## Training Details

| Component | Choice | Params |
|---|---|---|
| Vision encoder | DINOv2-Small (last block unfrozen) | ~3M trainable |
| Text encoder | precomputed on CPU, freed after | 0 on GPU |
| State encoder | 3-layer MLP, hidden=256 | ~0.15M trainable |
| Fusion | Cross-attention transformer, 4 layers, 8 heads | ~3.5M trainable |
| Diffusion head | DDPM, T=64, hidden=512, cond_proj MLP | ~3.5M trainable |
| **Total trainable** | | **~10M / 100M** |

Optimizer: AdamW, lr=1e-4, cosine decay, weight_decay=1e-4  
Gradient clipping: max_norm=1.0  
Dataset: 1000 episodes × 5 tasks × ~80 avg steps → ~400k `(obs, action_chunk)` pairs

---

## Patch Notes

### v0.4 — March 19, 2026
- **Camera fix** — identified default `behindGripper` camera as primary failure cause for contact tasks. Moving to `corner2` which shows full workspace with better depth cues
- **Resolution bump** — 84×84 → 128×128 for better object visibility
- **CLI rollout** — `--mode render|save_video|headless`, `--tasks`, `--episodes`, `--max_steps` args
- **Video recording** — `python rollout.py --mode save_video` saves per-episode mp4s to `videos/<task>/`
- **Multi-task rollout** — evaluates list of tasks in one run, prints per-task and overall summary table
- **`task_config.py`** — single source of truth for task instructions and difficulty tiers
- **`check_episode_lengths.py`** — utility to measure expert policy step distributions per task

### v0.3 — March 18, 2026
- **Modular encoder registry** — all encoders swappable via string config, no code changes needed
- **New vision encoders** — EfficientNet-B0, MobileNetV3-Small, DINOv2-Small
- **New text encoders** — SmolLM2-135M, BERT-Tiny
- **Base classes** — `BaseVisionEncoder`, `BaseTextEncoder`, `BaseStateEncoder` ABCs for clean extension
- **Precomputed text tokens** — text encoder runs once on CPU at startup, freed from GPU entirely
- **Multi-task training** — `ConcatDataset` over per-task folders, per-batch text token lookup by task_id
- **Dynamic `n_tokens`** — vision encoders auto-compute token count from input resolution
- **cuDNN + TF32 flags** — `cudnn.benchmark=True` and TF32 enabled for Ampere GPUs
- **obs_horizon=3** — adds acceleration signal on top of velocity

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
- [x] Observation history (obs_horizon=3)
- [x] Cross-attention fusion
- [x] WandB logging
- [x] HuggingFace push
- [x] Modular encoder registry
- [x] EfficientNet, MobileNet, DINOv2 vision encoders
- [x] SmolLM2, BERT-Tiny text encoders
- [x] Precomputed text tokens — zero text encoder memory on GPU
- [x] Multi-task data collection + joint training (MT5)
- [x] Video recording rollout
- [x] CLI rollout arguments
- [ ] Fix camera angle — recollect at 128×128 with corner2 (in progress)
- [ ] MT5 results with correct camera
- [ ] MT-10 full suite
- [ ] Flow matching action head
- [ ] CLIP vision + text encoder pair
- [ ] SigLIP vision encoder
- [ ] R3M manipulation-specific encoder
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