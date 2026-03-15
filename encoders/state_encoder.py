import torch
import torch.nn as nn

class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),   # was 128
            nn.SiLU(),                   # was ReLU
            nn.Linear(256, 256),         # extra layer
            nn.SiLU(),
            nn.Linear(256, d_model),
            nn.SiLU(),
        )

        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.layernorm(x)
        # make it a token so it matches vision/text tokens
        x = x.unsqueeze(1)   # (B,1,d_model)

        return x