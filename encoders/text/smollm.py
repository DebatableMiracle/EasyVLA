import torch
import torch.nn as nn
from transformers import AutoModel
from encoders.text.base import BaseTextEncoder


class TextEncoderSmolLM(BaseTextEncoder):
    def __init__(self, d_model=256):
        super().__init__(d_model)

        self.backbone = AutoModel.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.float32   # force float32 to match rest of model
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, d_model)
        self.layernorm  = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state          # (B, L, hidden_size)
        x = self.projection(x)
        x = self.layernorm(x)
        return x                     # (B, L, d_model)