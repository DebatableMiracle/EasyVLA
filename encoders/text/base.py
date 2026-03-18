from abc import ABC, abstractmethod
import torch.nn as nn


class BaseTextEncoder(ABC, nn.Module):
    """
    All text encoders must inherit this.
    Input:  input_ids (B, L), attention_mask (B, L)
    Output: (B, L, d_model) — token embeddings
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(self, input_ids, attention_mask):
        pass