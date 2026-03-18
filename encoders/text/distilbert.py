import torch.nn as nn
from transformers import DistilBertModel
from encoders.text.base import BaseTextEncoder


class TextEncoderDistilbert(BaseTextEncoder):
    def __init__(self, d_model=256):
        super().__init__(d_model)
        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(768, d_model)
        self.layernorm  = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        x = self.projection(x)
        x = self.layernorm(x)
        return x   # (B, L, d_model)