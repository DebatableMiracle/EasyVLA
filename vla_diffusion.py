import torch
import torch.nn as nn

from encoders.vision_encoder import VisionEncoderResnet18
from encoders.text_encoder import TextEncoderDistilbert
from encoders.state_encoder import StateEncoderMLP
from fusion import FusionTransformer
from action_head.diffusion_head import DiffusionHead, DiffusionConfig


class VlaDiffusion(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, diffusion_steps=16, action_horizon=16):
        super().__init__()

        self.action_horizon = action_horizon

        self.vision_encoder = VisionEncoderResnet18(d_model=d_model)
        self.text_encoder   = TextEncoderDistilbert(d_model=d_model)
        self.state_encoder  = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion         = FusionTransformer(d_model=d_model)

        cfg = DiffusionConfig(
            T=diffusion_steps,
            action_dim=action_dim,
            action_horizon=action_horizon,  # NEW
            cond_dim=d_model,
        )
        self.diffusion_head = DiffusionHead(cfg)

        # cache text tokens since instruction doesn't change within an episode
        self._text_cache = {}

    def encode_observations(self, image, input_ids, attention_mask, state):
        key = input_ids.sum().item()
        if key not in self._text_cache:
            self._text_cache[key] = self.text_encoder(input_ids, attention_mask).detach()

        text_tokens = self._text_cache[key].expand(image.size(0), -1, -1)
        vision_tokens = self.vision_encoder(image)
        state_tokens  = self.state_encoder(state)
        
        return self.fusion(vision_tokens, text_tokens, state_tokens)


    def loss(self, image, input_ids, attention_mask, state, action):
        # action: (B, action_horizon, action_dim)
        cond = self.encode_observations(image, input_ids, attention_mask, state)
        return self.diffusion_head.loss(action, cond)

    def act(self, image, input_ids, attention_mask, state):
        cond = self.encode_observations(image, input_ids, attention_mask, state)
        return self.diffusion_head.sample(cond)  # (B, action_horizon, action_dim)