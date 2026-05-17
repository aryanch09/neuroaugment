from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp_min(1e-4)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
        opt = torch.optim.LBFGS([self.log_temperature], max_iter=max_iter, line_search_fn="strong_wolfe")
        loss_fn = nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = loss_fn(self(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.temperature.detach())


def calibration_curve(y_true: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if np.any(m):
            accs.append(np.mean(y_true[m]))
            confs.append(np.mean(confidence[m]))
    return np.asarray(confs), np.asarray(accs)
