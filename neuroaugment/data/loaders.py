from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroaugment.core.augmenter import Augmenter
from neuroaugment.data.dataset import PhysioDataset


def make_loader(
    dataset: PhysioDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Standard supervised DataLoader for PhysioDataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ContrastiveDataset(Dataset):
    """Wraps PhysioDataset to emit (view1, view2, label) triples.

    Each call to __getitem__ runs Augmenter.apply_pair() to produce a
    causally-structured DRI view and a DRI+PLP view of the same sample.
    This is the core data pipeline for NeuroAugment SSL pretraining.

    Parameters
    ----------
    dataset   : base PhysioDataset
    augmenter : Augmenter with apply_pair() support (defaults to causal pipeline)
    """

    def __init__(self, dataset: PhysioDataset, augmenter: Optional[Augmenter] = None):
        self.dataset = dataset
        if augmenter is None:
            from neuroaugment.ssl.trainer import _default_augmenter
            augmenter = _default_augmenter()
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        x = item["x"].numpy()  # (T, C)
        (v1, m1), (v2, m2) = self.augmenter.apply_pair(x, sample_idx=idx)
        return {
            "x1": torch.as_tensor(v1, dtype=torch.float32),
            "x2": torch.as_tensor(v2, dtype=torch.float32),
            "y": item["y"],
            "meta1": m1,
            "meta2": m2,
            "view_type1": m1.get("view_type", "DRI"),
            "view_type2": m2.get("view_type", "DRI+PLP"),
        }


def make_contrastive_loader(
    dataset: PhysioDataset,
    augmenter: Optional[Augmenter] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """DataLoader that emits causally-structured contrastive view pairs."""
    return DataLoader(
        ContrastiveDataset(dataset, augmenter),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def make_federated_loaders(
    dataset: PhysioDataset,
    subjects: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> dict[int, DataLoader]:
    """Split dataset by subject and return per-subject DataLoaders for FL simulation."""
    loaders: dict[int, DataLoader] = {}
    for subj in np.unique(subjects):
        mask = subjects == subj
        subset = dataset._subset(np.where(mask)[0])
        loaders[int(subj)] = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loaders
