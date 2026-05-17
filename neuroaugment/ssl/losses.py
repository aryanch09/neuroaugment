from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE / NT-Xent loss (Chen et al., SimCLR 2020)."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def invariance_loss(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(h1, dim=-1), F.normalize(h2, dim=-1))


def causal_consistency_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE between predicted and target causal intervention deltas.

    Args:
        pred_delta: Predicted change in embedding space due to a causal intervention.
        target_delta: Ground-truth delta derived from the causal generative model.
        mask: Optional boolean/float mask selecting which dimensions or samples to include.
        reduction: 'mean' (default) or 'sum'.
    """
    err = (pred_delta - target_delta).pow(2)
    if mask is not None:
        err = err * mask.to(err.dtype)
        return err.sum() / (mask.to(err.dtype).sum().clamp_min(1))
    return err.mean() if reduction == "mean" else err.sum()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> torch.Tensor:
    """VICReg: Variance-Invariance-Covariance Regularization (Bardes et al., 2022).

    Avoids representational collapse without requiring negative pairs. Well-suited
    for small-batch biosignal pretraining where InfoNCE degrades.
    """
    N, D = z1.shape
    inv = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()) / 2
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (N - 1)
    cov2 = (z2c.T @ z2c) / (N - 1)
    off1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
    off2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
    cov = (off1 + off2) / D
    return sim_coeff * inv + std_coeff * var + cov_coeff * cov


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_off: float = 0.005) -> torch.Tensor:
    """Barlow Twins loss (Zbontar et al., 2021).

    Drives the cross-correlation matrix of the two views toward identity,
    explicitly penalising redundancy between embedding dimensions.
    """
    N = z1.shape[0]
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-4)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-4)
    C = (z1n.T @ z2n) / N
    diag = torch.diagonal(C)
    on_diag = (diag - 1).pow(2).sum()
    off_diag = C.pow(2).sum() - diag.pow(2).sum()
    return on_diag + lambda_off * off_diag
