import torch.nn as nn
from transformers import CLIPTextModel
from encoders.text.base import BaseTextEncoder


class TextEncoderCLIP(BaseTextEncoder):
    """
    CLIP text tower. Must use the same checkpoint as VisionEncoderCLIP
    so vision and text tokens are aligned in the same embedding space.
    That alignment is the whole point of using CLIP as a pair.

    Default: openai/clip-vit-base-patch32
    """

    CLIP_HIDDEN = 512  # CLIP text hidden dim for ViT-B/32

    def __init__(self, d_model=256, checkpoint="openai/clip-vit-base-patch32"):
        super().__init__(d_model)

        self.backbone = CLIPTextModel.from_pretrained(checkpoint)

        # freeze the backbone entirely 
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(self.CLIP_HIDDEN, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, L), attention_mask: (B, L)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state   # (B, L, 512)
        x = self.projection(x)      # (B, L, d_model)
        x = self.layernorm(x)
        return x                    # (B, L, d_model)