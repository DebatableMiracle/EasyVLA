# EasyVLA
##current state

```
I have finished working with Encoders, fusion and diffusion action head. The only thing left is to create a train script over MetaWorld dataset (which I might have to collect using their expert policy). 
```


# Vision-Language-Action (VLA) Policy from Scratch

This project implements a **compact Vision-Language-Action (VLA) policy** that learns to control a robot using **visual observations, language instructions, and robot state**.

The model follows the general architecture used in modern robotics systems:

```
vision + language + state → multimodal reasoning → action
```

The goal of this project is to build a **minimal yet representative VLA architecture** that can be trained on simulated robotics tasks.

---

# Architecture

The model processes three modalities:

```
image observation
language instruction
robot state
```

These inputs are encoded, fused with a transformer, and used to generate actions through a **diffusion policy head**.

Overall pipeline:

```
image
text instruction
robot state
↓
encoders
↓
fusion transformer
↓
context vector
↓
diffusion policy head
↓
robot action
```

---

# Model Components

## Vision Encoder

Visual observations are encoded using a **pretrained ResNet18 backbone**.

The network extracts spatial visual features and projects them into the model embedding space.

```
image → ResNet18 → vision tokens
```

Using pretrained weights allows the model to leverage features such as:

* edges
* shapes
* object boundaries

without training perception from scratch.

---

## Language Encoder

Language instructions are encoded using **DistilBERT**.

The model converts natural language instructions such as:

```
"push the block to the goal"
```

into a dense embedding that represents the task intent.

DistilBERT is frozen during training to reduce compute while still providing strong semantic representations.

```
text → DistilBERT → language tokens
```

---

## State Encoder

Robot state information (e.g. joint positions, velocities, gripper status) is encoded using a small **MLP**.

```
state → MLP → state token
```

This provides the policy with information about the robot’s current configuration.

---

## Multimodal Fusion Transformer

The encoded modalities are combined using a **Transformer encoder**.

Tokens from vision, language, and state are concatenated and processed jointly.

```
vision tokens
text tokens
state token
↓
Transformer
↓
context representation
```

A learned **CLS token** aggregates information across all modalities to produce a single context vector.

This allows the model to learn relationships such as:

* language ↔ visual objects
* robot state ↔ scene configuration
* instruction ↔ spatial features

---

## Diffusion Policy Head

Actions are generated using a **diffusion-based policy**.

Instead of predicting actions directly, the policy learns to **iteratively denoise actions** conditioned on the multimodal context.

Training objective:

```
predict noise added to expert actions
```

During inference:

```
noise → iterative denoising → action
```

This approach allows the model to represent **smooth and multimodal action distributions**.

---

# Implementation Status

Currently implemented:

* Vision encoder (ResNet18)
* Language encoder (DistilBERT)
* Robot state encoder
* Multimodal fusion transformer
* Diffusion policy head
* Full VLA policy architecture

---

# Next Steps

The next stage of the project is to train the policy using demonstrations from **MetaWorld**.

Planned workflow:

```
MetaWorld environment
↓
expert policy
↓
collect demonstrations
↓
train VLA policy
↓
evaluate robot behavior
```

Initial experiments will focus on single-task learning using the **push-v2** environment.

---

# Project Structure

```
models/
  vision_encoder.py
  text_encoder.py
  state_encoder.py
  fusion_transformer.py
  diffusion_head.py
  vla_policy.py
```

Upcoming modules:

```
data/
  collect_demos.py
  dataset.py

train.py
rollout.py
```

---

# Goals

This project aims to:

* understand VLA architectures
* build a minimal but representative robotics policy
* explore diffusion policies for robot control
* experiment with language-conditioned robot behavior

---

# References

Key inspiration for this implementation includes research on:

* Vision-Language-Action models
* Diffusion policies for robotics
* multimodal transformer architectures


