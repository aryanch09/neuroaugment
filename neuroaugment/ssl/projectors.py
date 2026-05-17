from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class MLPProjector(nn.Module):
    """2-layer MLP projection head with optional BatchNorm.

    BatchNorm between layers is critical for contrastive training stability
    (Chen et al., SimCLR 2020; Zbontar et al., Barlow Twins 2021).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        hidden_dim: Optional[int] = None,
        use_bn: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([nn.ReLU(), nn.Linear(hidden_dim, out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
