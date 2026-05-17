from __future__ import annotations

import numpy as np
from scipy import signal


def resample_signal(X: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = int(round(X.shape[0] * target_fs / orig_fs))
    return signal.resample(X, n, axis=0)


def window_signal(X: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    X = np.asarray(X)
    if X.shape[0] < window_size:
        raise ValueError("window_size exceeds signal length")
    return np.stack([X[i : i + window_size] for i in range(0, X.shape[0] - window_size + 1, stride)])


def zscore_normalize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + eps)


def robust_normalize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(X, axis=0, keepdims=True)
    mad = np.median(np.abs(X - med), axis=0, keepdims=True)
    return (X - med) / (1.4826 * mad + eps)
