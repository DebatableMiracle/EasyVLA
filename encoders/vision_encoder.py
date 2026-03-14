import torch
import torch.nn as nn
import torchvision.models as models

class VisionEncoderResnet18(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone[-2].parameters():
            p.requires_grad = True
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        #the idea is that we freeze all the layers except the last one understands and learns most of the task level features!

    def forward(self, x):
        x=self.backbone(x)
        # convert spatial grid → tokens
        x = x.flatten(2)          # (B,512,49)
        x = x.transpose(1,2)      # (B,49,512)
        x=self.projection(x)
        x=self.layernorm(x)
        return x
