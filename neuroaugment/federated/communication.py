from __future__ import annotations

import numpy as np


def topk_sparsify(arr: np.ndarray, k_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    flat = np.asarray(arr).reshape(-1)
    k = max(1, int(round(k_fraction * flat.size)))
    idx = np.argpartition(np.abs(flat), -k)[-k:]
    return idx.astype(np.int64), flat[idx].copy(), arr.shape


def topk_desparsify(indices: np.ndarray, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    flat = np.zeros(int(np.prod(shape)), dtype=values.dtype)
    flat[indices] = values
    return flat.reshape(shape)


def uniform_quantize(arr: np.ndarray, num_bits: int = 8) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(arr, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    levels = 2**num_bits - 1
    q = np.round((arr - mn) / (mx - mn + 1e-12) * levels).astype(np.uint8 if num_bits <= 8 else np.uint16)
    return q, mn, mx


def uniform_dequantize(q: np.ndarray, mn: float, mx: float, num_bits: int = 8) -> np.ndarray:
    return q.astype(float) / (2**num_bits - 1) * (mx - mn) + mn
