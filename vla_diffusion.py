import torch
import torch.nn as nn

from encoders.registry import build_vision_encoder, build_text_encoder, build_state_encoder
from fusion import FusionTransformer
from action_head.diffusion_head import DiffusionHead, DiffusionConfig


class VlaDiffusion(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        d_model        = 256,
        diffusion_steps= 64,
        action_horizon = 8,
        obs_horizon    = 2,
        vision_encoder = "resnet18",
        text_encoder   = "distilbert",
        state_encoder  = "mlp",
    ):
        super().__init__()

        self.action_horizon = action_horizon
        self.obs_horizon    = obs_horizon

        self.vision_encoder = build_vision_encoder(vision_encoder, d_model, obs_horizon)
        self.text_encoder   = build_text_encoder(text_encoder, d_model)
        self.state_encoder  = build_state_encoder(state_encoder, state_dim, d_model)
        self.fusion         = FusionTransformer(d_model=d_model)

        cfg = DiffusionConfig(
            T              = diffusion_steps,
            action_dim     = action_dim,
            action_horizon = action_horizon,
            cond_dim       = d_model,
        )
        self.diffusion_head = DiffusionHead(cfg)

    def encode_observations(self, image, text_tokens, state):
        # text_tokens: (1, L, d_model) precomputed — expand to batch
        txt           = text_tokens.expand(image.size(0), -1, -1).to(image.device)
        vision_tokens = self.vision_encoder(image)
        state_tokens  = self.state_encoder(state)
        return self.fusion(vision_tokens, txt, state_tokens)

    def loss(self, image, text_tokens, state, action):
        cond = self.encode_observations(image, text_tokens, state)
        return self.diffusion_head.loss(action, cond)

    def act(self, image, text_tokens, state):
        cond = self.encode_observations(image, text_tokens, state)
        return self.diffusion_head.sample(cond)  # (B, action_horizon, action_dim)