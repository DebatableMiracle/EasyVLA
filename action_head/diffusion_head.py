import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    T: int = 32
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    action_dim: int = 4
    action_horizon: int = 16    # we predict H actions at once resulting in a more stable policy. 
    cond_dim: int = 256


def make_beta_schedule(cfg: DiffusionConfig):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half_dim, device=device)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class ActionDenoiseModel(nn.Module):
    def __init__(self, cfg: DiffusionConfig, time_emb_dim=32, hidden_dim=256):  # wider for chunk
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        chunk_dim = cfg.action_dim * cfg.action_horizon  
        in_dim = chunk_dim + time_emb_dim + cfg.cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_dim),
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_emb(t)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        return self.net(x)


class DiffusionHead(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoise_model = ActionDenoiseModel(cfg)
        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def loss(self, actions, cond):
        # actions: (B, action_horizon, action_dim)
        B = actions.size(0)
        actions_flat = actions.view(B, -1)              # (B, action_horizon * action_dim)
        t = torch.randint(0, self.cfg.T, (B,), device=actions.device)
        noise = torch.randn_like(actions_flat)
        x_t = self.q_sample(actions_flat, t, noise)
        eps_pred = self.denoise_model(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond):
        self.eval()
        B = cond.size(0)
        chunk_dim = self.cfg.action_dim * self.cfg.action_horizon
        x_t = torch.randn(B, chunk_dim, device=cond.device)

        for t_step in reversed(range(self.cfg.T)):
            t = torch.full((B,), t_step, device=cond.device, dtype=torch.long)
            eps_pred = self.denoise_model(x_t, t, cond)
            beta_t      = self.betas[t_step]
            alpha_bar_t = self.alpha_bar[t_step]
            alpha_t     = self.alphas[t_step]
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            if t_step > 0:
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * torch.randn_like(x_t)
            else:
                x_t = x0_pred

        return x_t.view(B, self.cfg.action_horizon, self.cfg.action_dim)  # (B, H, action_dim)