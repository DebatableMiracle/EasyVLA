from encoders.vision.resnet       import VisionEncoderResnet18
from encoders.vision.efficientnet import VisionEncoderEfficientNet
from encoders.vision.dinov2       import VisionEncoderDINOv2
from encoders.text.distilbert     import TextEncoderDistilbert
from encoders.text.smollm         import TextEncoderSmolLM
from encoders.text.bert_tiny      import TextEncoderBERTTiny
from encoders.state.mlp           import StateEncoderMLP
from encoders.vision.clip import VisionEncoderCLIP
from encoders.text.clip import TextEncoderCLIP

# in build_vision_encoder:

# in build_text_encoder:


VISION_ENCODERS = {
    "resnet18":     VisionEncoderResnet18,
    "efficientnet": VisionEncoderEfficientNet,
    "dinov2":       VisionEncoderDINOv2,
    "clip": lambda d_model, obs_horizon: VisionEncoderCLIP(d_model, obs_horizon),

    # "clip":    VisionEncoderCLIP,    # coming soon
    # "siglip":  VisionEncoderSigLIP,  # coming soon
    # "r3m":     VisionEncoderR3M,     # coming soon
    # "mobilenet":    VisionEncoderMobileNet, 
}

TEXT_ENCODERS = {
    "distilbert": TextEncoderDistilbert,
    "smollm":     TextEncoderSmolLM,
    "bert_tiny":  TextEncoderBERTTiny,
    "clip": lambda d_model: TextEncoderCLIP(d_model),
    # "clip": TextEncoderCLIP,   # coming soon
    # "t5":   TextEncoderT5,     # coming soon
}

STATE_ENCODERS = {
    "mlp": StateEncoderMLP,
}


def build_vision_encoder(name: str, d_model: int, obs_horizon: int):
    if name not in VISION_ENCODERS:
        raise ValueError(f"Unknown vision encoder '{name}'. Available: {list(VISION_ENCODERS)}")
    return VISION_ENCODERS[name](d_model=d_model, obs_horizon=obs_horizon)


def build_text_encoder(name: str, d_model: int):
    if name not in TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder '{name}'. Available: {list(TEXT_ENCODERS)}")
    return TEXT_ENCODERS[name](d_model=d_model)


def build_state_encoder(name: str, state_dim: int, d_model: int):
    if name not in STATE_ENCODERS:
        raise ValueError(f"Unknown state encoder '{name}'. Available: {list(STATE_ENCODERS)}")
    return STATE_ENCODERS[name](state_dim=state_dim, d_model=d_model)