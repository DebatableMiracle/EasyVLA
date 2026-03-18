import torch.nn as nn
from transformers import AutoModel
from encoders.text.base import BaseTextEncoder


class TextEncoderBERTTiny(BaseTextEncoder):
    """
    BERT-Tiny — 4.4M params, 2 layers, hidden=128.
    Fastest text encoder option. Good enough for simple task instructions.
    """
    def __init__(self, d_model=256):
        super().__init__(d_model)

        self.backbone = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(128, d_model)  # bert-tiny hidden = 128
        self.layernorm  = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state         # (B, L, 128)
        x = self.projection(x)
        x = self.layernorm(x)
        return x                    # (B, L, d_model)