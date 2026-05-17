from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal biosignals (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2 + d_model % 2])[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _ResBlock1d(nn.Module):
    """1-D residual block with GELU activation for temporal feature extraction."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class TemporalCNNEncoder(nn.Module):
    """Temporal 1-D CNN encoder with residual connections for physiological time series."""

    def __init__(self, in_channels: int, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res1 = _ResBlock1d(hidden_dim, hidden_dim, kernel=5)
        self.res2 = _ResBlock1d(hidden_dim, latent_dim, kernel=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input (B,T,C)")
        h = self.stem(x.transpose(1, 2))
        h = self.res1(h)
        h = self.res2(h)
        return self.pool(h).squeeze(-1)


class TransformerEncoder(nn.Module):
    """Transformer encoder with sinusoidal positional encoding for temporal biosignals."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            latent_dim,
            nhead,
            dim_feedforward=4 * latent_dim,
            batch_first=True,
            dropout=dropout,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.proj(x))
        return self.norm(self.encoder(h).mean(dim=1))
