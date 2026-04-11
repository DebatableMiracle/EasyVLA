import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from encoders.vision.base import BaseVisionEncoder


class VisionEncoderCLIP(BaseVisionEncoder):
    """
    CLIP ViT-B/32 vision tower.
    Returns patch tokens (not CLS) so fusion has spatial context.
    ViT-B/32 at 224x224 gives 49 patch tokens.

    Important: use the same CLIP checkpoint as your text encoder.
    Default: openai/clip-vit-base-patch32
    """

    CLIP_HIDDEN = 768  # ViT-B hidden dim

    def __init__(self, d_model=256, obs_horizon=2, checkpoint="openai/clip-vit-base-patch32"):
        super().__init__(d_model, obs_horizon)

        self.backbone = CLIPVisionModel.from_pretrained(checkpoint)

        # CLIP expects 3-channel input, but we stack obs_horizon frames.
        # We replace the patch embedding conv to accept obs_horizon*3 channels,
        # averaging the pretrained weights across the new channel groups.
        original_conv = self.backbone.vision_model.embeddings.patch_embedding
        pretrained_w = original_conv.weight.clone()  # (embed_dim, 3, P, P)

        new_conv = nn.Conv2d(
            obs_horizon * 3,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight = nn.Parameter(
                pretrained_w.repeat(1, obs_horizon, 1, 1) / obs_horizon
            )

        self.backbone.vision_model.embeddings.patch_embedding = new_conv

        # freeze everything except last 2 transformer layers
        for p in self.backbone.parameters():
            p.requires_grad = False
        for layer in self.backbone.vision_model.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.projection = nn.Linear(self.CLIP_HIDDEN, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    @property
    def n_tokens(self):
        return 49  # ViT-B/32 at 224x224

    def forward(self, x):
        # x: (B, obs_horizon*3, H, W)
        out = self.backbone(pixel_values=x, output_hidden_states=False)

        # last_hidden_state: (B, 1+49, 768) — index 0 is CLS, rest are patch tokens
        patch_tokens = out.last_hidden_state[:, 1:, :]  # (B, 49, 768)

        patch_tokens = self.projection(patch_tokens)    # (B, 49, d_model)
        patch_tokens = self.layernorm(patch_tokens)
        return patch_tokens                             # (B, 49, d_model)