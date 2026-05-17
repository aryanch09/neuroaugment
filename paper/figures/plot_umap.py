from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal

from neuroaugment.evaluation.visualization import plot_umap_or_pca
from neuroaugment.simulator import Simulator


MODALITIES = ("ecg", "eeg", "imu")


def _bandpower(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    f, pxx = signal.welch(x, fs=fs, axis=0, nperseg=min(256, x.shape[0]))
    mask = (f >= low) & (f <= high)
    return np.trapz(pxx[mask], f[mask], axis=0)


def _features(x: np.ndarray, fs: float) -> np.ndarray:
    q = np.quantile(x, [0.05, 0.5, 0.95], axis=0).reshape(-1)
    stats = np.r_[x.mean(axis=0), x.std(axis=0), q]
    bands = np.r_[
        _bandpower(x, fs, 0.5, 4.0),
        _bandpower(x, fs, 4.0, 8.0),
        _bandpower(x, fs, 8.0, 14.0),
        _bandpower(x, fs, 14.0, 40.0),
    ]
    corr = np.corrcoef(x.T)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return np.nan_to_num(np.r_[stats, np.log1p(bands), upper], nan=0.0)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    X = np.asarray(data["embeddings"] if "embeddings" in data else data["X"], dtype=float)
    y = np.asarray(data["labels"] if "labels" in data else data.get("y", np.zeros(len(X))), dtype=int)
    sites = np.asarray(data["sites"] if "sites" in data else np.zeros(len(X)), dtype=int)
    return X, y, sites


def _synthetic_embedding(n_per_modality: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats, labels, sites = [], [], []
    for label, modality in enumerate(MODALITIES):
        fs = 100.0 if modality == "imu" else 250.0
        for site in range(3):
            sim = Simulator(modality=modality, T=640, C=3, fs=fs, seed=seed + 100 * label + site)
            X, _, _ = sim.sample_batch(max(1, n_per_modality // 3))
            site_shift = 1.0 + 0.035 * site
            for x in X:
                feats.append(_features(site_shift * x, fs))
                labels.append(label)
                sites.append(site)
    return np.vstack(feats), np.asarray(labels), np.asarray(sites)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/umap.png")
    parser.add_argument("--input", type=Path, default=None, help="Optional NPZ with embeddings/X, labels/y, sites.")
    parser.add_argument("--method", choices=["auto", "umap", "pca"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-modality", type=int, default=60)
    args = parser.parse_args()
    if args.input is not None:
        X, y, sites = _load_npz(args.input)
    else:
        X, y, sites = _synthetic_embedding(args.n_per_modality, args.seed)
    plot_umap_or_pca(
        X,
        y,
        args.out,
        sites=sites,
        label_names=MODALITIES,
        method=args.method,
        seed=args.seed,
        title="NeuroAugment latent space separates physiology while preserving site shift",
    )


if __name__ == "__main__":
    main()
