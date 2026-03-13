from transformers import DistilBertModel
import torch.nn as nn

class TextEncoderDistilbert(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        self.backbone = DistilbertModel.from_pretrained("distilbert-base-uncased")

        #freeze all the layers
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(768, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):

        x = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state        # (B, L, 768)

        x = self.projection(x)     # (B, L, d_model)
        x = self.layernorm(x)

        return x