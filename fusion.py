import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, d_model = 256, n_heads = 4, n_layers =2, dropout=0.1):
        super().__init__()
        
        # learned CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            n_head=n_heads,
            dim_feedforward= 4*d_model,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
            )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            )
        
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, vision_tokens, text_tokens, state_tokens):
        B = vision_tokens.size(0)

        #concatenate the tokens
        tokens = torch.cat([vision_tokens, text_tokens, state_tokens], dim=1)

        #prepend the cls token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        #multimodal reasoning
        x = self.transformer(tokens)

        # take CLS token as our final context vector
        x = x[:, 0]

        context = self.layernorm(x)
        return context

        