import torch
import torch.nn as nn
from encoders.vision.base import BaseVisionEncoder


class VisionEncoderDINOv2(BaseVisionEncoder):
    """
    DINOv2-Small — 22M params, patch_size=14, hidden=384.
    Best spatial features for manipulation — used in OpenVLA.
    obs_horizon frames merged into 3 channels via learned 1x1 conv.
    """
    def __init__(self, d_model=256, obs_horizon=2):
        super().__init__(d_model, obs_horizon)

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            verbose=False
        )

        # DINOv2 expects 3 channels — learned 1x1 conv merges stacked frames
        self.channel_proj = nn.Conv2d(
            obs_horizon * 3, 3,
            kernel_size=1, bias=False
        )
        # init as averaging across frames
        with torch.no_grad():
            self.channel_proj.weight.fill_(1.0 / (obs_horizon * 3))

        for p in self.backbone.parameters():
            p.requires_grad = False
        # unfreeze last transformer block
        for p in self.backbone.blocks[-1].parameters():
            p.requires_grad = True

        self.projection = nn.Linear(384, d_model)  # DINOv2-S hidden = 384
        self.layernorm  = nn.LayerNorm(d_model)

    @property
    def n_tokens(self):
        return 256   # 224x224 / patch_size=14 → 16x16 = 256 tokens
                     # 84x84  / patch_size=14 → 6x6  = 36 tokens — override if needed

    def forward(self, x):
        x = self.channel_proj(x)                              # (B, 3, H, W)
        x = self.backbone.get_intermediate_layers(x, n=1)[0] # (B, N, 384)
        x = self.projection(x)
        x = self.layernorm(x)
        return x                                              # (B, N, d_model)