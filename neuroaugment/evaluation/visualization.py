from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PALETTE = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]


def _prepare_out(out: Union[str, Path]) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0f172a",
            "axes.titlecolor": "#0f172a",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#cbd5e1",
            "grid.linewidth": 0.6,
        }
    )


def compute_2d_embedding(
    embeddings: np.ndarray,
    method: str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, str, Optional[np.ndarray]]:
    X = StandardScaler().fit_transform(np.asarray(embeddings, dtype=float))
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("embeddings must have shape (N,D) with N >= 2")
    if method in {"auto", "umap"}:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(20, max(2, X.shape[0] // 5)),
                min_dist=0.12,
                metric="cosine",
                random_state=seed,
            )
            return reducer.fit_transform(X), "UMAP", None
        except Exception:
            if method == "umap":
                raise
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), "PCA", pca.explained_variance_ratio_


def _confidence_ellipse(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 4:
        return
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(vals, 1e-12))
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.10,
        linewidth=1.2,
    )
    ax.add_patch(ellipse)


def plot_umap_or_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out: Union[str, Path],
    sites: Optional[np.ndarray] = None,
    label_names: Optional[Sequence[str]] = None,
    method: str = "auto",
    seed: int = 0,
    title: str = "Latent physiology map",
) -> Path:
    out = _prepare_out(out)
    _set_paper_style()
    coords, method_name, explained = compute_2d_embedding(embeddings, method, seed)
    labels = np.asarray(labels)
    sites = np.asarray(sites) if sites is not None else np.zeros(len(labels), dtype=int)
    markers = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for i, lab in enumerate(np.unique(labels)):
        mask = labels == lab
        color = PALETTE[i % len(PALETTE)]
        for j, site in enumerate(np.unique(sites)):
            smask = mask & (sites == site)
            if not np.any(smask):
                continue
            name = label_names[int(lab)] if label_names is not None and int(lab) < len(label_names) else f"class {lab}"
            legend = name if j == 0 else None
            ax.scatter(
                coords[smask, 0],
                coords[smask, 1],
                s=26,
                marker=markers[j % len(markers)],
                c=color,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.35,
                label=legend,
            )
        _confidence_ellipse(ax, coords[mask], color)

    subtitle = method_name
    if explained is not None:
        subtitle += f" ({100 * explained.sum():.1f}% variance)"
    ax.set_title(title)
    ax.set_xlabel(f"{subtitle} 1")
    ax.set_ylabel(f"{subtitle} 2")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="best", ncols=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    return out


def plot_metric_bars(
    metrics: dict[str, float],
    out: Union[str, Path],
    errors: Optional[dict[str, float]] = None,
    title: str = "Downstream performance",
) -> Path:
    out = _prepare_out(out)
    _set_paper_style()
    names = list(metrics)
    values = np.asarray([metrics[n] for n in names], dtype=float)
    yerr = None if errors is None else np.asarray([errors.get(n, 0.0) for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.bar(names, values, yerr=yerr, color=colors, edgecolor="#0f172a", linewidth=0.5, capsize=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, min(1.05, max(0.2, float(values.max()) + 0.14)))
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    return out


def plot_privacy_utility(
    eps: np.ndarray,
    utility: np.ndarray,
    out: Union[str, Path],
    method: str = "DP-SGD",
    title: str = "Privacy-utility frontier",
) -> Path:
    out = _prepare_out(out)
    _set_paper_style()
    eps = np.asarray(eps, dtype=float)
    utility = np.asarray(utility, dtype=float)
    order = np.argsort(eps)
    eps, utility = eps[order], utility[order]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(eps, utility, marker="o", color=PALETTE[0], linewidth=2.0, label=method)
    ax.fill_between(eps, utility.min() - 0.02, utility, color=PALETTE[0], alpha=0.10)
    knee = int(np.argmax(utility - 0.04 * np.log1p(eps)))
    ax.scatter([eps[knee]], [utility[knee]], s=80, color=PALETTE[1], zorder=5, label="selected operating point")
    ax.annotate(
        f"eps={eps[knee]:.2f}\nscore={utility[knee]:.2f}",
        xy=(eps[knee], utility[knee]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("privacy budget epsilon (log scale)")
    ax.set_ylabel("utility score")
    ax.set_title(title)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    return out
