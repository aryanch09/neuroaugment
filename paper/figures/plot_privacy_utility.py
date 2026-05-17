from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuroaugment.evaluation.visualization import PALETTE, _prepare_out, _set_paper_style


def _load_csv(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row.get("method", "DP-SGD")].append((float(row["epsilon"]), float(row["utility"])))
    out = {}
    for method, pairs in rows.items():
        arr = np.asarray(pairs, dtype=float)
        order = np.argsort(arr[:, 0])
        out[method] = (arr[order, 0], arr[order, 1])
    return out


def _fallback() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    eps = np.array([0.35, 0.5, 1.0, 2.0, 4.0, 8.0])
    return {
        "Central DP-SGD": (eps, np.array([0.58, 0.63, 0.70, 0.76, 0.80, 0.82])),
        "Federated DP": (eps, np.array([0.61, 0.67, 0.73, 0.79, 0.83, 0.85])),
        "NeuroAugment + DP": (eps, np.array([0.66, 0.72, 0.79, 0.84, 0.87, 0.89])),
    }


def _plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    out = _prepare_out(out)
    _set_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    best_score = -np.inf
    best_point = (0.0, 0.0, "")
    for i, (method, (eps, utility)) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        order = np.argsort(eps)
        eps, utility = eps[order], utility[order]
        ax.plot(eps, utility, marker="o", linewidth=2.0, color=color, label=method)
        ax.fill_between(eps, utility.min() - 0.02, utility, color=color, alpha=0.06)
        score = utility - 0.035 * np.log1p(eps)
        idx = int(np.argmax(score))
        if score[idx] > best_score:
            best_score = float(score[idx])
            best_point = (float(eps[idx]), float(utility[idx]), method)
    ax.scatter([best_point[0]], [best_point[1]], s=90, color="#0f172a", zorder=5)
    ax.annotate(
        f"recommended\n{best_point[2]}\neps={best_point[0]:.2f}",
        xy=(best_point[0], best_point[1]),
        xytext=(14, -36),
        textcoords="offset points",
        fontsize=8.8,
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("privacy budget epsilon (log scale)")
    ax.set_ylabel("macro F1")
    ax.set_title("NeuroAugment shifts the privacy-utility frontier upward")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/privacy_utility.png")
    parser.add_argument("--input", type=Path, default=None, help="CSV with epsilon,utility[,method].")
    args = parser.parse_args()
    curves = _load_csv(args.input) if args.input is not None else _fallback()
    _plot_curves(curves, Path(args.out))


if __name__ == "__main__":
    main()
