from abc import ABC, abstractmethod
import torch.nn as nn


class BaseVisionEncoder(ABC, nn.Module):
    """
    All vision encoders must inherit this.
    Input:  (B, obs_horizon*3, H, W) — stacked RGB frames
    Output: (B, N, d_model)          — patch/spatial tokens
    """
    def __init__(self, d_model: int, obs_horizon: int):
        super().__init__()
        self.d_model     = d_model
        self.obs_horizon = obs_horizon

    @abstractmethod
    def forward(self, x):
        # x: (B, obs_horizon*3, H, W)
        # returns: (B, N, d_model)  N = number of tokens
        pass

    @property
    @abstractmethod
    def n_tokens(self) -> int:
        # how many tokens this encoder outputs per image
        # resnet18 @ 224x224 → 49, efficientnet → 49, dinov2 → 256 etc
        pass