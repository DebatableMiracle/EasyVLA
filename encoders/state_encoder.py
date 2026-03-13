import torch
import torch.nn as nn

class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
            nn.ReLU(),
        )

        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.layernorm(x)
        # make it a token so it matches vision/text tokens
        x = x.unsqueeze(1)   # (B,1,d_model)

        return x