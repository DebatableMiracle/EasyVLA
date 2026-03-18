import torch
import torch.nn as nn
import torchvision.models as models
from encoders.vision.base import BaseVisionEncoder


class VisionEncoderEfficientNet(BaseVisionEncoder):
    def __init__(self, d_model=256, obs_horizon=2):
        super().__init__(d_model, obs_horizon)

        net = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # grab weights before replacing first conv
        pretrained_w = net.features[0][0].weight.clone()

        net.features[0][0] = nn.Conv2d(
            obs_horizon * 3, 32,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        with torch.no_grad():
            net.features[0][0].weight = nn.Parameter(
                pretrained_w.repeat(1, obs_horizon, 1, 1) / obs_horizon
            )

        self.backbone = net.features  # (B, 1280, 7, 7) for 224x224

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

        self.projection = nn.Linear(1280, d_model)
        self.layernorm  = nn.LayerNorm(d_model)

    @property
    def n_tokens(self):
        return 49

    def forward(self, x):
        x = self.backbone(x)       # (B, 1280, 7, 7)
        x = x.flatten(2)           # (B, 1280, 49)
        x = x.transpose(1, 2)      # (B, 49, 1280)
        x = self.projection(x)
        x = self.layernorm(x)
        return x                   # (B, 49, d_model)