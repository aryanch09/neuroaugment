"""SSL baselines for comparison with NeuroAugment.

Each baseline returns a pretrained encoder (nn.Module) that maps (B,T,C)→(B,D),
ready to be evaluated with the protocols in benchmarks.protocols.

Baselines
---------
RandomInit       — untrained encoder (lower bound)
SimCLRBaseline   — SimCLR with Gaussian noise augmentation (naive, no causal structure)
SupervisedBaseline — fully supervised encoder (upper bound)
NeuroAugBaseline — NeuroAugment causal SSL (our method)
"""
from __future__ import annotations

import copy
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neuroaugment.ssl.encoders import TemporalCNNEncoder, TransformerEncoder
from neuroaugment.ssl.losses import barlow_twins_loss, info_nce_loss, vicreg_loss
from neuroaugment.ssl.projectors import MLPProjector
from neuroaugment.ssl.trainer import SSLTrainer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_encoder(
    in_channels: int,
    latent_dim: int,
    encoder: str,
    device: torch.device,
) -> nn.Module:
    enc: nn.Module = (
        TemporalCNNEncoder(in_channels, latent_dim)
        if encoder == "cnn"
        else TransformerEncoder(in_channels, latent_dim)
    )
    return enc.to(device)


# ---------------------------------------------------------------------------
# Random Init (lower bound)
# ---------------------------------------------------------------------------

class RandomInit:
    """Randomly initialised encoder — establishes the lower-bound baseline.

    Demonstrates how much of the performance is attributable to the
    architecture rather than the learned representations.
    """

    name = "RandomInit"

    def __init__(self, in_channels: int, latent_dim: int = 64, encoder: str = "cnn"):
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder_type = encoder

    def pretrain(
        self,
        X: np.ndarray,
        device: Optional[str] = None,
        **_kwargs,
    ) -> nn.Module:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return _make_encoder(self.in_channels, self.latent_dim, self.encoder_type, dev)


# ---------------------------------------------------------------------------
# SimCLR with Gaussian noise (naive augmentation — the strawman)
# ---------------------------------------------------------------------------

class SimCLRBaseline:
    """SimCLR (Chen et al., 2020) with i.i.d. Gaussian noise augmentation.

    This is the *naive* contrastive baseline that ignores the causal
    structure of biosignals — what NeuroAugment is designed to improve upon.
    Views are created by adding Gaussian noise, without any physiological
    or device-level invariance.
    """

    name = "SimCLR-GaussNoise"

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        encoder: str = "cnn",
        noise_std: float = 0.1,
        temperature: float = 0.2,
    ):
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder_type = encoder
        self.noise_std = noise_std
        self.temperature = temperature

    def pretrain(
        self,
        X: np.ndarray,
        steps: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> nn.Module:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encoder = _make_encoder(self.in_channels, self.latent_dim, self.encoder_type, dev)
        projector = MLPProjector(self.latent_dim, self.latent_dim, use_bn=True).to(dev)
        params = list(encoder.parameters()) + list(projector.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        rng = np.random.default_rng(0)
        X_t = torch.as_tensor(X, dtype=torch.float32)

        encoder.train(); projector.train()
        for _ in range(steps):
            idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=len(X) < batch_size)
            batch = X_t[idx].to(dev)
            x1 = batch + self.noise_std * torch.randn_like(batch)
            x2 = batch + self.noise_std * torch.randn_like(batch)
            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))
            loss = info_nce_loss(z1, z2, self.temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()

        return encoder


# ---------------------------------------------------------------------------
# Supervised (upper bound)
# ---------------------------------------------------------------------------

class SupervisedBaseline:
    """Fully supervised training — establishes the performance upper bound.

    Encoder is trained end-to-end with cross-entropy on all available labels.
    """

    name = "Supervised"

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        latent_dim: int = 64,
        encoder: str = "cnn",
    ):
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder_type = encoder

    def pretrain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> nn.Module:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encoder = _make_encoder(self.in_channels, self.latent_dim, self.encoder_type, dev)
        head = nn.Linear(self.latent_dim, self.n_classes).to(dev)
        params = list(encoder.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        ds = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        iter_loader = iter(loader)

        encoder.train(); head.train()
        for _ in range(steps):
            try:
                xb, yb = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                xb, yb = next(iter_loader)
            xb, yb = xb.to(dev), yb.to(dev)
            loss = nn.functional.cross_entropy(head(encoder(xb)), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()

        return encoder


# ---------------------------------------------------------------------------
# NeuroAugment (our method)
# ---------------------------------------------------------------------------

class NeuroAugBaseline:
    """NeuroAugment causal SSL — the proposed method.

    Uses Augmenter.apply_pair() to generate DRI / DRI+PLP view pairs that
    respect the causal generative model X = D[G(P)] + N.

    Supports three SSL objectives: infonce, vicreg, barlow.
    """

    name = "NeuroAugment"

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 64,
        encoder: str = "cnn",
        loss: Literal["infonce", "vicreg", "barlow"] = "vicreg",
        temperature: float = 0.2,
    ):
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder_type = encoder
        self.loss = loss
        self.temperature = temperature

    def pretrain(
        self,
        X: np.ndarray,
        steps: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> nn.Module:
        trainer = SSLTrainer(
            in_channels=self.in_channels,
            latent_dim=self.latent_dim,
            encoder=self.encoder_type,
            loss=self.loss,
            temperature=self.temperature,
            device=device,
        )
        trainer.fit_arrays(X, steps=steps, batch_size=batch_size)
        return trainer.encoder


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

class NeuroAugDRIOnly(NeuroAugBaseline):
    """Ablation: DRI augmentations only (no physiological label-preserving ops).

    Demonstrates the added value of including PLP augmentations in the view
    generation pipeline.
    """

    name = "NeuroAugment-DRIonly"

    def pretrain(self, X: np.ndarray, steps: int = 500, batch_size: int = 64, lr: float = 1e-3, device: Optional[str] = None) -> nn.Module:
        from neuroaugment.core.augmenter import Augmenter
        from neuroaugment.core.operators import channel_crosstalk, colored_noise_addition, device_frequency_response, channel_dropout

        # Only device-level (DRI) augmentations, no temporal causal masking (PLP)
        aug = Augmenter(
            [device_frequency_response, channel_crosstalk, colored_noise_addition, channel_dropout],
            op_probabilities=[0.8, 0.6, 1.0, 0.3],
            seed=42,
        )
        trainer = SSLTrainer(
            in_channels=self.in_channels,
            latent_dim=self.latent_dim,
            encoder=self.encoder_type,
            loss=self.loss,
            temperature=self.temperature,
            device=device,
            augmenter=aug,
        )
        trainer.fit_arrays(X, steps=steps, batch_size=batch_size)
        return trainer.encoder


class NeuroAugInfoNCE(NeuroAugBaseline):
    """Ablation: NeuroAugment with InfoNCE instead of VICReg."""
    name = "NeuroAugment-InfoNCE"

    def __init__(self, in_channels: int, latent_dim: int = 64, encoder: str = "cnn", temperature: float = 0.2):
        super().__init__(in_channels, latent_dim, encoder, loss="infonce", temperature=temperature)


# ---------------------------------------------------------------------------
# Registry of all baselines for the runner
# ---------------------------------------------------------------------------

ALL_BASELINES = [RandomInit, SimCLRBaseline, NeuroAugDRIOnly, NeuroAugInfoNCE, NeuroAugBaseline]
