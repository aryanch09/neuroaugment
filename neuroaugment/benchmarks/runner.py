"""Unified benchmark runner for NeuroAugment.

Runs every combination of (dataset, baseline, protocol) and prints a
LaTeX-ready results table.  Designed for reproducible ablation studies.

Usage
-----
    from neuroaugment.benchmarks.runner import BenchmarkRunner
    runner = BenchmarkRunner(data_root="/data/neuroaugment", device="cuda")
    table = runner.run(datasets=["mitbih", "ucihar"], steps=1000)
    print(table.to_latex())
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from neuroaugment.benchmarks.baselines import (
    ALL_BASELINES,
    NeuroAugBaseline,
    NeuroAugDRIOnly,
    NeuroAugInfoNCE,
    RandomInit,
    SimCLRBaseline,
    SupervisedBaseline,
)
from neuroaugment.benchmarks.protocols import (
    CrossSubjectLOSO,
    EvalResult,
    FewShotFinetune,
    LinearEval,
)
from neuroaugment.data.public_datasets import DATASETS, get_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset default configs
# ---------------------------------------------------------------------------

_DATASET_CFG: dict[str, dict] = {
    "mitbih":   {"window_s": 10.0, "stride_s": 5.0,  "modality": "ecg",  "fs": 360},
    "ptbxl":    {"modality": "ecg",  "fs": 100},
    "eegmmidb": {"window_s": 2.0,  "stride_s": 1.0,  "modality": "eeg",  "fs": 160, "task": "motor_imagery_LR"},
    "bciiv2a":  {"window_s": 2.0,  "stride_s": 1.0,  "modality": "eeg",  "fs": 250},
    "ucihar":   {"modality": "imu", "fs": 50},
    "pamap2":   {"window_s": 2.0,  "stride_s": 1.0,  "modality": "imu",  "fs": 100},
    "wisdm":    {"window_s": 2.0,  "stride_s": 1.0,  "modality": "imu",  "fs": 20},
}


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

class ResultsTable:
    """Accumulates EvalResult entries and renders as Markdown or LaTeX."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, result: EvalResult, baseline_name: str) -> None:
        row = asdict(result)
        row["baseline"] = baseline_name
        self._rows.append(row)

    def to_markdown(self) -> str:
        header = "| Dataset | Baseline | Protocol | AUROC | F1 | ECE | Acc |"
        sep    = "|---------|----------|----------|-------|----|-----|-----|"
        lines  = [header, sep]
        for r in self._rows:
            lines.append(
                f"| {r['dataset']} | {r['baseline']} | {r['protocol']} "
                f"| {r['auroc']:.4f} | {r['f1']:.4f} "
                f"| {r['ece']:.4f} | {r['accuracy']:.4f} |"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Render as LaTeX booktabs table (requires \\usepackage{booktabs})."""
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{NeuroAugment benchmark results}",
            r"\label{tab:results}",
            r"\begin{tabular}{llllcccc}",
            r"\toprule",
            r"Dataset & Baseline & Protocol & AUROC & F1 & ECE & Acc \\",
            r"\midrule",
        ]
        for r in self._rows:
            lines.append(
                rf"{r['dataset']} & {r['baseline']} & {r['protocol']} "
                rf"& {r['auroc']:.4f} & {r['f1']:.4f} "
                rf"& {r['ece']:.4f} & {r['accuracy']:.4f} \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._rows, f, indent=2)
        logger.info("Results saved → %s", path)

    def cross_site_gap_summary(self) -> str:
        gaps = [(r["baseline"], r["dataset"], r["extra"].get("cross_site_gap", float("nan")))
                for r in self._rows if "cross_site_gap" in r.get("extra", {})]
        if not gaps:
            return "No LOSO cross-site gap data."
        lines = ["Cross-site AUROC gap (↓ better):"]
        for bl, ds, gap in gaps:
            lines.append(f"  {bl:30s}  {ds:12s}  Δ={gap:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run all baselines × protocols on specified datasets.

    Parameters
    ----------
    data_root  : root directory containing all dataset sub-folders
    device     : torch device string
    latent_dim : encoder latent dimension (shared across all baselines)
    encoder    : 'cnn' or 'transformer'
    steps      : SSL pretraining steps
    batch_size : pretraining batch size
    lr         : pretraining learning rate
    few_shot_fracs : label fractions for FewShot protocol
    run_loso   : whether to run LOSO (slow for large datasets)
    max_loso_subjects : cap LOSO at this many subjects for speed
    seed       : global random seed
    """

    def __init__(
        self,
        data_root: str,
        device: Optional[str] = None,
        latent_dim: int = 64,
        encoder: str = "cnn",
        steps: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        few_shot_fracs: Optional[list[float]] = None,
        run_loso: bool = True,
        max_loso_subjects: int = 20,
        seed: int = 42,
    ) -> None:
        import torch
        self.data_root = data_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.few_shot_fracs = few_shot_fracs or [0.01, 0.05, 0.10, 1.0]
        self.run_loso = run_loso
        self.max_loso_subjects = max_loso_subjects
        self.seed = seed

    def _load(self, name: str) -> tuple:
        """Load train and test splits for a dataset."""
        cls = get_dataset(name)
        cfg = _DATASET_CFG.get(name, {})
        logger.info("Loading %s …", name)

        load_kwargs = {k: v for k, v in cfg.items() if k not in ("modality", "fs")}

        try:
            X_tr, y_tr, s_tr, m_tr = cls.load(self.data_root, split="train", **load_kwargs)
            X_te, y_te, s_te, m_te = cls.load(self.data_root, split="test", **load_kwargs)
        except FileNotFoundError as exc:
            logger.error("Dataset %s not found: %s — skipping.", name, exc)
            return None

        logger.info(
            "%s loaded: train=%d test=%d  channels=%d  T=%d",
            name, len(X_tr), len(X_te), X_tr.shape[2], X_tr.shape[1],
        )
        return X_tr, y_tr, s_tr, X_te, y_te, s_te

    def _pretrain(self, name: str, baseline, X_tr, y_tr):
        """Pretrain a baseline encoder and return it."""
        import torch
        n_channels = X_tr.shape[2]
        n_classes = int(y_tr.max()) + 1

        t0 = time.time()
        if isinstance(baseline, RandomInit):
            enc = baseline.pretrain(X_tr, device=self.device)
        elif isinstance(baseline, SupervisedBaseline):
            enc = baseline.pretrain(X_tr, y_tr, steps=self.steps, batch_size=self.batch_size, lr=self.lr, device=self.device)
        else:
            enc = baseline.pretrain(X_tr, steps=self.steps, batch_size=self.batch_size, lr=self.lr, device=self.device)
        logger.info("  Pretrained %s on %s in %.1f s", baseline.name, name, time.time() - t0)
        return enc

    def run(
        self,
        datasets: Optional[list[str]] = None,
        out_dir: Optional[str] = None,
    ) -> ResultsTable:
        """Run all benchmarks and return a ResultsTable.

        Parameters
        ----------
        datasets : list of dataset keys (default: all registered datasets)
        out_dir  : if set, saves results.json and results_table.md here
        """
        target_datasets = datasets or list(DATASETS.keys())
        table = ResultsTable()

        for ds_name in target_datasets:
            loaded = self._load(ds_name)
            if loaded is None:
                continue
            X_tr, y_tr, s_tr, X_te, y_te, s_te = loaded
            n_channels = X_tr.shape[2]
            n_classes = int(max(y_tr.max(), y_te.max())) + 1

            # -------- build baselines for this dataset --------
            baselines = [
                RandomInit(n_channels, self.latent_dim, self.encoder),
                SimCLRBaseline(n_channels, self.latent_dim, self.encoder),
                NeuroAugDRIOnly(n_channels, self.latent_dim, self.encoder),
                NeuroAugInfoNCE(n_channels, self.latent_dim, self.encoder),
                NeuroAugBaseline(n_channels, self.latent_dim, self.encoder, loss="vicreg"),
                SupervisedBaseline(n_channels, n_classes, self.latent_dim, self.encoder),
            ]

            for baseline in baselines:
                logger.info("=== %s | %s ===", ds_name, baseline.name)
                enc = self._pretrain(ds_name, baseline, X_tr, y_tr)

                # --- Linear Eval ---
                try:
                    ev = LinearEval(enc, n_epochs=100, lr=0.01)
                    ev.fit(X_tr, y_tr)
                    res = ev.evaluate(X_te, y_te, dataset_name=ds_name)
                    table.add(res, baseline.name)
                    logger.info("  LinearEval: %s", res)
                except Exception as exc:
                    logger.warning("  LinearEval failed: %s", exc)

                # --- Few-Shot ---
                try:
                    fs = FewShotFinetune(enc, label_fracs=self.few_shot_fracs, n_epochs=50, seed=self.seed)
                    fs_results = fs.run(X_tr, y_tr, X_te, y_te, dataset_name=ds_name)
                    for r in fs_results:
                        table.add(r, baseline.name)
                    logger.info("  FewShot: %d runs", len(fs_results))
                except Exception as exc:
                    logger.warning("  FewShot failed: %s", exc)

                # --- LOSO ---
                if self.run_loso:
                    try:
                        X_all = np.concatenate([X_tr, X_te])
                        y_all = np.concatenate([y_tr, y_te])
                        s_all = np.concatenate([s_tr, s_te])
                        loso = CrossSubjectLOSO(enc, n_epochs_head=100, max_subjects=self.max_loso_subjects)
                        _, summary = loso.run(X_all, y_all, s_all, dataset_name=ds_name)
                        table.add(summary, baseline.name)
                        logger.info("  LOSO: %s", summary)
                    except Exception as exc:
                        logger.warning("  LOSO failed: %s", exc)

        # -------- Save outputs --------
        if out_dir:
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            table.save_json(str(p / "results.json"))
            (p / "results_table.md").write_text(table.to_markdown())
            (p / "results_table.tex").write_text(table.to_latex())
            logger.info("Results written to %s", out_dir)

        return table


# ---------------------------------------------------------------------------
# Privacy-utility sweep
# ---------------------------------------------------------------------------

def privacy_utility_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_multipliers: Optional[list[float]] = None,
    steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    delta: float = 1e-5,
    device: Optional[str] = None,
    dataset_name: str = "unknown",
) -> list[dict]:
    """Sweep noise_multiplier values and report epsilon vs. AUROC.

    Returns a list of dicts with keys: noise_multiplier, epsilon, auroc, f1.
    """
    import torch
    from neuroaugment.privacy.accounting import rdp_epsilon
    from neuroaugment.privacy.dp_trainer import DPTrainer
    from neuroaugment.ssl.encoders import TemporalCNNEncoder
    from neuroaugment.ssl.losses import vicreg_loss
    from neuroaugment.ssl.projectors import MLPProjector
    from neuroaugment.ssl.trainer import SSLTrainer

    noise_multipliers = noise_multipliers or [0.5, 1.0, 1.5, 2.0, 3.0, 0.0]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_channels = X_train.shape[2]
    q = batch_size / len(X_train)
    results = []

    for nm in noise_multipliers:
        if nm == 0.0:
            # No DP — baseline
            trainer = SSLTrainer(n_channels, loss="vicreg", device=device)
            trainer.fit_arrays(X_train, steps=steps, batch_size=batch_size)
            enc = trainer.encoder
            eps = float("inf")
        else:
            # DP training
            trainer = SSLTrainer(n_channels, loss="vicreg", device=device)
            dp = DPTrainer(
                nn.Sequential(trainer.encoder, trainer.projector),
                trainer.opt,
                max_grad_norm=1.0,
                noise_multiplier=nm,
            )
            rng_np = np.random.default_rng(0)
            X_t = torch.as_tensor(X_train, dtype=torch.float32)
            for step_i in range(steps):
                idx = rng_np.choice(len(X_train), size=min(batch_size, len(X_train)), replace=False)
                (v1_arr, _), (v2_arr, _) = trainer._augmenter.apply_pair(X_train[idx[0]], sample_idx=step_i)
                x1 = torch.as_tensor(np.stack([v1_arr] * len(idx)), dtype=torch.float32).to(dev)
                x2 = torch.as_tensor(np.stack([v2_arr] * len(idx)), dtype=torch.float32).to(dev)
                z1 = trainer.projector(trainer.encoder(x1))
                z2 = trainer.projector(trainer.encoder(x2))
                loss = vicreg_loss(z1, z2)
                dp.step(loss)
            enc = trainer.encoder
            eps = rdp_epsilon(sample_rate=q, noise_multiplier=nm, steps=steps, delta=delta)

        ev = LinearEval(enc, n_epochs=50)
        ev.fit(X_train, y_train)
        res = ev.evaluate(X_test, y_test, dataset_name=dataset_name)
        results.append({
            "noise_multiplier": nm,
            "epsilon": eps,
            "auroc": res.auroc,
            "f1": res.f1,
            "accuracy": res.accuracy,
        })
        logger.info("DP sweep  nm=%.1f  ε=%.2f  AUROC=%.4f", nm, eps, res.auroc)

    return results
