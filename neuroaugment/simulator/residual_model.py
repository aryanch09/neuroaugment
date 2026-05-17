from __future__ import annotations

import torch
from torch import nn


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim + cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim + cond_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(torch.cat([x, cond], dim=-1))
        return h.chunk(2, dim=-1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, cond], dim=-1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, cond)
        return self.decode(self.reparameterize(mu, logvar), cond), mu, logvar

    @staticmethod
    def loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        rec = torch.nn.functional.mse_loss(recon, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + 1e-3 * kld
