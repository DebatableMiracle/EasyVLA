import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()

        # self attention over visual + text context
        self_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.context_encoder = nn.TransformerEncoder(self_layer, num_layers=n_layers)

        # cross attention — state/action queries attend to visual+text context
        cross_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.cross_attention = nn.TransformerDecoder(cross_layer, num_layers=n_layers)

        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, vision_tokens, text_tokens, state_tokens):
        # build context from vision + text (keys/values)
        context = torch.cat([vision_tokens, text_tokens], dim=1)  # (B, 49+L, d_model)
        context = self.context_encoder(context)

        # state tokens query the visual+text context (queries)
        # this is where "where is the target relative to my arm?" gets answered
        x = self.cross_attention(
            tgt=state_tokens,    # (B, 1, d_model) — queries
            memory=context,      # (B, 49+L, d_model) — keys/values
        )                        # (B, 1, d_model)

        return self.layernorm(x.squeeze(1))  # (B, d_model)
