from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Literal, Optional

from neuroaugment.core.augmenter import Augmenter
from neuroaugment.core.operators import (
    channel_crosstalk,
    channel_dropout,
    colored_noise_addition,
    device_frequency_response,
    temporal_causal_masking,
)
from neuroaugment.ssl.encoders import TemporalCNNEncoder, TransformerEncoder
from neuroaugment.ssl.losses import barlow_twins_loss, info_nce_loss, vicreg_loss
from neuroaugment.ssl.projectors import MLPProjector

LossFn = Literal["infonce", "vicreg", "barlow"]


def _default_augmenter(seed: int = 42) -> Augmenter:
    """Causal augmentation pipeline ordered by DRI / PLP split.

    Device-relative-invariant (DRI) operators are listed first so that
    Augmenter.apply_pair() can separate views by augmentation family.
    """
    return Augmenter(
        [
            device_frequency_response,
            channel_crosstalk,
            colored_noise_addition,
            channel_dropout,
            temporal_causal_masking,
        ],
        op_probabilities=[0.8, 0.6, 1.0, 0.3, 0.5],
        seed=seed,
    )


class SSLTrainer:
    """Self-supervised pre-training for physiological time-series encoders.

    Views are generated with Augmenter.apply_pair(), which separates
    Device-Relative-Invariant (DRI) from DRI+Physiological-Label-Preserving
    (DRI+PLP) augmentations — the core causal invariance structure of NeuroAugment.

    Args:
        in_channels: Number of input channels (C in the causal model).
        latent_dim: Encoder output dimensionality.
        lr: Initial learning rate for AdamW.
        encoder: 'cnn' (TemporalCNNEncoder) or 'transformer' (TransformerEncoder).
        loss: SSL objective — 'infonce', 'vicreg', or 'barlow'.
        temperature: InfoNCE temperature (ignored for vicreg / barlow).
        device: Torch device string; auto-selects CUDA if available.
        augmenter: Custom Augmenter; defaults to the standard causal pipeline.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        lr: float = 1e-3,
        encoder: str = "cnn",
        loss: LossFn = "infonce",
        temperature: float = 0.2,
        device: Optional[str] = None,
        augmenter: Optional[Augmenter] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        enc: nn.Module = (
            TemporalCNNEncoder(in_channels, latent_dim)
            if encoder == "cnn"
            else TransformerEncoder(in_channels, latent_dim)
        )
        self.encoder = enc.to(self.device)
        self.projector = MLPProjector(latent_dim, latent_dim, use_bn=True).to(self.device)
        self._params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.opt = torch.optim.AdamW(self._params, lr=lr, weight_decay=1e-4)
        self.scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
        self.loss_fn: LossFn = loss
        self.temperature = temperature
        self._augmenter = augmenter or _default_augmenter()

    def _build_scheduler(self, total_steps: int) -> None:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total_steps, eta_min=1e-6
        )

    def _compute_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.loss_fn == "vicreg":
            return vicreg_loss(z1, z2)
        if self.loss_fn == "barlow":
            return barlow_twins_loss(z1, z2)
        return info_nce_loss(z1, z2, self.temperature)

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """One gradient step on a pre-built view pair (B,T,C) tensors."""
        x1 = x1.to(self.device).float()
        x2 = x2.to(self.device).float()
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        loss = self._compute_loss(z1, z2)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, max_norm=1.0)
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.detach().cpu())

    def fit_arrays(
        self,
        X: np.ndarray,
        steps: int = 10,
        batch_size: int = 8,
        noise_std: float = 0.0,
    ) -> list[float]:
        """Train on raw numpy array X of shape (N, T, C).

        Views are generated via Augmenter.apply_pair(), producing a DRI view and a
        DRI+PLP view per sample, preserving the causal structure of the generative model.

        Args:
            noise_std: Additional i.i.d. Gaussian jitter (0 by default; prefer augmenter ops).
        """
        if self.scheduler is None:
            self._build_scheduler(steps)
        rng = np.random.default_rng(0)
        losses: list[float] = []
        for step_i in range(steps):
            idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=len(X) < batch_size)
            views1, views2 = [], []
            for j, si in enumerate(idx):
                global_idx = step_i * batch_size + j
                (v1_arr, _), (v2_arr, _) = self._augmenter.apply_pair(X[si], sample_idx=global_idx)
                views1.append(torch.as_tensor(v1_arr, dtype=torch.float32))
                views2.append(torch.as_tensor(v2_arr, dtype=torch.float32))
            x1 = torch.stack(views1)
            x2 = torch.stack(views2)
            if noise_std > 0:
                x1 = x1 + noise_std * torch.randn_like(x1)
                x2 = x2 + noise_std * torch.randn_like(x2)
            losses.append(self.step(x1, x2))
        return losses
