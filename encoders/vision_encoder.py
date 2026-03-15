import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoderResnet18(nn.Module):
    def __init__(self, d_model=256, obs_horizon=2):
        super().__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        resnet.conv1 = nn.Conv2d(
            obs_horizon * 3, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            pretrained_w = models.resnet18(weights="IMAGENET1K_V1").conv1.weight  # (64,3,7,7)
            resnet.conv1.weight = nn.Parameter(
                pretrained_w.repeat(1, obs_horizon, 1, 1) / obs_horizon  # (64, 3*H, 7, 7)
            )

        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # freeze all except last two layers
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[-1].parameters():  # layer4
            p.requires_grad = True
        for p in self.backbone[-2].parameters():  # layer3
            p.requires_grad = True

        self.projection = nn.Linear(512, d_model)
        self.layernorm  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.backbone(x)       # (B, 512, 7, 7)
        x = x.flatten(2)           # (B, 512, 49)
        x = x.transpose(1, 2)      # (B, 49, 512)
        x = self.projection(x)     # (B, 49, d_model)
        x = self.layernorm(x)
        return x