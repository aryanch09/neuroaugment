from __future__ import annotations

import numpy as np
from scipy import signal


def ar_noise(T: int, C: int, ar_coeffs: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((T, C))
    for c in range(C):
        out[:, c] = signal.lfilter([1.0], np.r_[1.0, -np.asarray(ar_coeffs)[:2]], rng.normal(0, std, T))
    return out


def baseline_wander(T: int, C: int, fs: float, amplitude: float, freq: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) / fs
    phase = rng.uniform(0, 2 * np.pi, C)
    return amplitude * np.sin(2 * np.pi * freq * t[:, None] + phase[None, :])
