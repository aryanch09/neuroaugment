from __future__ import annotations

import torch


class DPTrainer:
    """Differentially-private training wrapper (simplified DP-SGD).

    Implements the gradient-noise mechanism from Abadi et al. (2016).
    Limitation: uses aggregate gradient clipping rather than true per-sample
    clipping.  True per-sample clipping requires ghost clipping (Li et al., 2022)
    or an Opacus-style backward hook; use the `opacus` library for
    publication-grade (epsilon, delta)-DP guarantees.

    For the privacy accounting that accompanies this trainer, use
    `neuroaugment.privacy.accounting.rdp_epsilon`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.steps = 0

    def step(self, loss: torch.Tensor) -> float:
        """Backward pass with gradient clipping and Gaussian noise injection."""
        self.optimizer.zero_grad()
        loss.backward()
        # Clip aggregate gradient norm (approximates per-sample clip when batch=1
        # or when per-sample norms are roughly equal).
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        sigma = self.noise_multiplier * self.max_grad_norm
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad) * sigma)
        self.optimizer.step()
        self.steps += 1
        return float(loss.detach().cpu())

    @property
    def privacy_spent(self) -> dict:
        """Convenience accessor — returns the hyperparameters needed for accounting."""
        return {"steps": self.steps, "noise_multiplier": self.noise_multiplier, "max_grad_norm": self.max_grad_norm}
