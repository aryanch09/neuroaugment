from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred))


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (Naeini et al., 2015).

    Weighted average |accuracy - confidence| over equal-width confidence bins,
    where bin weight = fraction of samples in that bin.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    if probs.ndim == 2:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
    else:
        conf = probs
        pred = (probs >= 0.5).astype(int)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if np.any(m):
            # m.mean() = bin_count / n_total — correct proportional weighting
            ece += float(m.mean() * abs(acc[m].mean() - conf[m].mean()))
    return ece


def cross_site_gap(site_scores: dict[str, float]) -> float:
    """Max minus min performance across sites — lower is more generalisable."""
    vals = np.asarray(list(site_scores.values()), dtype=float)
    return float(np.round(np.nanmax(vals) - np.nanmin(vals), 12))


def label_preservation_score(
    y_orig: np.ndarray,
    y_aug: np.ndarray,
) -> float:
    """Fraction of samples where augmentation preserves the ground-truth label.

    For biosignal augmentation, physiological label-preserving (PLP) operators
    should leave the event label unchanged.  A score close to 1.0 confirms that
    the augmentation respects the causal structure of the generative model.
    """
    y_orig = np.asarray(y_orig).ravel()
    y_aug = np.asarray(y_aug).ravel()
    if y_orig.shape != y_aug.shape:
        raise ValueError("y_orig and y_aug must have the same shape")
    return float((y_orig == y_aug).mean())


def augmentation_diversity_score(
    embeddings_orig: np.ndarray,
    embeddings_aug: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean cosine distance between original and augmented embeddings.

    Measures how much augmentations perturb the representation space.
    Good augmentations should be non-trivial (score > 0) while not
    destroying discriminative structure (score << 1 after training).

    Args:
        embeddings_orig: (N, D) encoder outputs for original samples.
        embeddings_aug:  (N, D) encoder outputs for augmented counterparts.
    """
    e1 = np.asarray(embeddings_orig, dtype=float)
    e2 = np.asarray(embeddings_aug, dtype=float)
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + eps)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + eps)
    cos_sim = (e1 * e2).sum(axis=1)
    return float(1.0 - cos_sim.mean())
