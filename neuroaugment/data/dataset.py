from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PhysioDataset(Dataset):
    """In-memory dataset for physiological time-series windows.

    Parameters
    ----------
    signals : (N, T, C) float array
    labels  : (N,) int array (or (N, K) for multi-label)
    metas   : list of per-sample metadata dicts
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metas: Optional[list[dict]] = None,
    ):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = (
            np.zeros(len(self.signals), dtype=np.int64)
            if labels is None
            else np.asarray(labels)
        )
        self.metas = metas or [{} for _ in range(len(self.signals))]
        if len(self.signals) != len(self.labels):
            raise ValueError("signals and labels must have the same first dimension")

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.from_numpy(self.signals[idx]).float(),
            "y": torch.as_tensor(self.labels[idx]),
            "meta": self.metas[idx],
        }

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def split(self, val_frac: float = 0.1, seed: int = 0) -> tuple["PhysioDataset", "PhysioDataset"]:
        """Random train/val split."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))
        n_val = max(1, int(round(val_frac * len(self))))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return self._subset(train_idx), self._subset(val_idx)

    def stratified_split(
        self, val_frac: float = 0.1, seed: int = 0
    ) -> tuple["PhysioDataset", "PhysioDataset"]:
        """Stratified train/val split that preserves class proportions."""
        rng = np.random.default_rng(seed)
        y = self.labels if self.labels.ndim == 1 else self.labels.argmax(axis=1)
        train_idx, val_idx = [], []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            rng.shuffle(ci)
            n_v = max(1, int(round(val_frac * len(ci))))
            val_idx.extend(ci[:n_v].tolist())
            train_idx.extend(ci[n_v:].tolist())
        return self._subset(np.asarray(train_idx)), self._subset(np.asarray(val_idx))

    def loso(self, subjects: np.ndarray):
        """Yield (train_dataset, test_dataset, held_out_subject) for LOSO-CV."""
        for held_out in np.unique(subjects):
            test_mask = subjects == held_out
            train_mask = ~test_mask
            yield (
                self._subset(np.where(train_mask)[0]),
                self._subset(np.where(test_mask)[0]),
                int(held_out),
            )

    def _subset(self, indices: np.ndarray) -> "PhysioDataset":
        return PhysioDataset(
            self.signals[indices],
            self.labels[indices],
            [self.metas[i] for i in indices],
        )
