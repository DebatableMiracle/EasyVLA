import torch

from encoders.vision_encoder import VisionEncoderResnet18
from encoders.text_encoder import TextEncoderDistilbert
from encoders.state_encoder import StateEncoderMLP
from fusion import FusionTransformer

class VlaDiffusion(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, diffusion_steps =16):
        super().__init__()

        #encoders
        self.vision_encoder = VisionEncoderResnet18(d_model = d_model)
        self.text_encoder = TextEncoderDistilbert(d_model = d_model)
        self.state_encoder = StateEncoderMLP(state_dim = state_dim, d_model = d_model)

        #fusion
        self.fusion = FusionTransformer(d_model = d_model)

        #diffusion
        self.diffusion_head = DiffusionHead(
            d_model = d_model,
            cond_dim = action_dim,
            T= diffusion_steps
            )
        

        def encode_observations(self, image, input_ids, attention_mask, state):
            vision_tokens = self.vision_encoder(image)

            text_tokens = self.text_encoder(input_ids, attention_mask)

            state_tokens = self.state_encoder(state)

            context = self.fusion(
                vision_tokens,
                text_tokens,
                state_tokens
            )

            return context

        def loss(self, image, input_ids, attention_mask, state, action):
            cond = self.encode_observations(image, input_ids, attention_mask, state)

            return self.diffusion_head.loss(action, cond)

        def act(
            self,
            image,
            input_ids,
            attention_mask,
            state,
        ):
            cond = self.encode_observations(image, input_ids, attention_mask, state)
            actions = self.diffusion_head.sample(cond)
            return actions

