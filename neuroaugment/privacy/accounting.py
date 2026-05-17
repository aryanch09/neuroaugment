from __future__ import annotations

import numpy as np
from typing import Optional


def compute_rdp(q: float, noise_multiplier: float, steps: int, orders: np.ndarray) -> np.ndarray:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    q = float(q)
    orders = np.asarray(orders, dtype=float)
    return steps * (q**2) * orders / (2 * noise_multiplier**2)


def get_privacy_spent(orders: np.ndarray, rdp: np.ndarray, delta: float) -> tuple[float, float]:
    eps = rdp - np.log(delta) / (orders - 1)
    idx = int(np.argmin(eps))
    return float(eps[idx]), float(orders[idx])


def rdp_epsilon(sample_rate: float, noise_multiplier: float, steps: int, delta: float = 1e-5, orders: Optional[np.ndarray] = None) -> float:
    orders = np.asarray(orders if orders is not None else np.arange(2, 64), dtype=float)
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    eps, _ = get_privacy_spent(orders, rdp, delta)
    return eps
