"""Evaluation protocols for SSL representation learning on biosignals.

Three standard protocols used in the biosignal SSL literature:

LinearEval       — freeze encoder, fit logistic regression on top
FewShotFinetune  — end-to-end fine-tune with a small labelled fraction
CrossSubjectLOSO — leave-one-subject-out generalisation test

All protocols accept a ``torch.nn.Module`` encoder that maps (B,T,C) → (B,D).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neuroaugment.evaluation.metrics import auroc_score, ece_score, f1_score_binary


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    protocol: str
    dataset: str
    auroc: float
    f1: float
    ece: float
    accuracy: float
    n_train: int
    n_test: int
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.protocol} | {self.dataset}] "
            f"AUROC={self.auroc:.4f}  F1={self.f1:.4f}  "
            f"ECE={self.ece:.4f}  Acc={self.accuracy:.4f}  "
            f"(n_train={self.n_train}, n_test={self.n_test})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(
    encoder: nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Pass (N,T,C) numpy array through encoder, return (N,D) numpy."""
    device = device or next(encoder.parameters()).device
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            out.append(encoder(batch).cpu().numpy())
    return np.concatenate(out)


def _binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, float]:
    """Return (auroc, f1, ece, accuracy) for binary or multiclass."""
    preds = probs.argmax(axis=1) if probs.ndim == 2 else (probs >= 0.5).astype(int)
    score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs.max(axis=1)
    acc = float((preds == y_true).mean())
    try:
        auroc = auroc_score(y_true, score)
    except Exception:
        auroc = float("nan")
    try:
        f1 = float(f1_score_binary(y_true, preds))
    except Exception:
        f1 = float("nan")
    ece = ece_score(y_true, probs if probs.ndim == 2 else np.stack([1 - probs, probs], axis=1))
    return auroc, f1, ece, acc


# ---------------------------------------------------------------------------
# Protocol 1: Linear Evaluation
# ---------------------------------------------------------------------------

class LinearEval:
    """Frozen encoder + trained linear head.

    Standard linear evaluation (LeCun et al.; SimCLR; MoCo):
    freeze the pretrained backbone, fit logistic regression
    on top of the embeddings using all available labelled training data.

    Parameters
    ----------
    encoder  : pretrained nn.Module mapping (B,T,C) → (B,D)
    n_epochs : epochs for linear head SGD
    lr       : learning rate
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
    ):
        self.encoder = encoder
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self._head: Optional[nn.Linear] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearEval":
        """Encode X_train and fit linear classifier."""
        device = next(self.encoder.parameters()).device
        Z = _encode(self.encoder, X_train, self.batch_size, device)
        n_classes = int(y_train.max()) + 1
        self._head = nn.Linear(Z.shape[1], n_classes).to(device)
        opt = torch.optim.SGD(self._head.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        ds = TensorDataset(
            torch.as_tensor(Z, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self._head.train()
        for _ in range(self.n_epochs):
            for zb, yb in loader:
                zb, yb = zb.to(device), yb.to(device)
                loss = nn.functional.cross_entropy(self._head(zb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "unknown",
    ) -> EvalResult:
        assert self._head is not None, "Call fit() first."
        device = next(self.encoder.parameters()).device
        Z = _encode(self.encoder, X_test, self.batch_size, device)
        self._head.eval()
        with torch.no_grad():
            logits = self._head(torch.as_tensor(Z, dtype=torch.float32).to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        auroc, f1, ece, acc = _binary_metrics(y_test, probs)
        return EvalResult(
            protocol="LinearEval",
            dataset=dataset_name,
            auroc=auroc,
            f1=f1,
            ece=ece,
            accuracy=acc,
            n_train=len(X_test),
            n_test=len(X_test),
        )


# ---------------------------------------------------------------------------
# Protocol 2: Few-Shot Fine-tuning
# ---------------------------------------------------------------------------

class FewShotFinetune:
    """End-to-end fine-tuning with a limited labelled fraction.

    Evaluates the encoder's sample efficiency — a key claim of causal
    augmentation: representations that respect the causal structure require
    fewer labels to fine-tune to a new task.

    Parameters
    ----------
    encoder       : pretrained nn.Module
    label_fracs   : fractions of training labels to sweep (e.g. [0.01, 0.05, 0.1, 1.0])
    n_epochs      : epochs per run
    lr            : learning rate (encoder)
    lr_head       : learning rate (head); defaults to 10× lr
    """

    def __init__(
        self,
        encoder: nn.Module,
        label_fracs: list[float] | None = None,
        n_epochs: int = 50,
        lr: float = 1e-4,
        lr_head: Optional[float] = None,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.encoder = encoder
        self.label_fracs = label_fracs or [0.01, 0.05, 0.10, 0.50, 1.0]
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_head = lr_head or (lr * 10)
        self.batch_size = batch_size
        self.seed = seed

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "unknown",
    ) -> list[EvalResult]:
        device = next(self.encoder.parameters()).device
        n_classes = int(y_train.max()) + 1
        rng = np.random.default_rng(self.seed)
        results = []

        for frac in self.label_fracs:
            n_use = max(n_classes, int(round(frac * len(X_train))))
            # Stratified subsample
            idx = []
            for c in range(n_classes):
                ci = np.where(y_train == c)[0]
                n_c = max(1, int(round(frac * len(ci))))
                idx.extend(rng.choice(ci, size=min(n_c, len(ci)), replace=False).tolist())
            idx = np.asarray(idx)
            X_sub, y_sub = X_train[idx], y_train[idx]

            # Clone encoder and add head
            import copy
            enc = copy.deepcopy(self.encoder).to(device)
            # Infer latent dim
            with torch.no_grad():
                dummy = torch.zeros(1, X_sub.shape[1], X_sub.shape[2], device=device)
                D = enc(dummy).shape[-1]
            head = nn.Linear(D, n_classes).to(device)
            params = [
                {"params": enc.parameters(), "lr": self.lr},
                {"params": head.parameters(), "lr": self.lr_head},
            ]
            opt = torch.optim.AdamW(params, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs)

            ds = TensorDataset(
                torch.as_tensor(X_sub, dtype=torch.float32),
                torch.as_tensor(y_sub, dtype=torch.long),
            )
            loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)

            enc.train(); head.train()
            for _ in range(self.n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = nn.functional.cross_entropy(head(enc(xb)), yb)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
                    opt.step()
                sched.step()

            # Evaluate
            enc.eval(); head.eval()
            Z_test = _encode(enc, X_test, device=device)
            with torch.no_grad():
                logits = head(torch.as_tensor(Z_test, dtype=torch.float32).to(device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            auroc, f1, ece, acc = _binary_metrics(y_test, probs)
            results.append(EvalResult(
                protocol=f"FewShot_{int(frac*100)}pct",
                dataset=dataset_name,
                auroc=auroc, f1=f1, ece=ece, accuracy=acc,
                n_train=len(X_sub), n_test=len(X_test),
                extra={"label_frac": frac},
            ))
        return results


# ---------------------------------------------------------------------------
# Protocol 3: Cross-Subject Leave-One-Subject-Out
# ---------------------------------------------------------------------------

class CrossSubjectLOSO:
    """Leave-one-subject-out cross-validation.

    Measures cross-subject generalisation — critical for clinical deployment
    and the cross-site gap metric.  Uses linear evaluation head for speed.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_epochs_head: int = 100,
        lr_head: float = 0.01,
        batch_size: int = 256,
        max_subjects: Optional[int] = None,
    ):
        self.encoder = encoder
        self.n_epochs_head = n_epochs_head
        self.lr_head = lr_head
        self.batch_size = batch_size
        self.max_subjects = max_subjects

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        dataset_name: str = "unknown",
    ) -> tuple[list[EvalResult], EvalResult]:
        """Run LOSO-CV and return per-subject results + aggregated summary.

        Returns
        -------
        per_subject : list of EvalResult (one per held-out subject)
        summary     : EvalResult with mean metrics + cross_site_gap in extra
        """
        unique = np.unique(subjects)
        if self.max_subjects and len(unique) > self.max_subjects:
            rng = np.random.default_rng(0)
            unique = rng.choice(unique, size=self.max_subjects, replace=False)

        per_subject = []
        for held_out in unique:
            test_mask = subjects == held_out
            train_mask = ~test_mask
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]
            if len(np.unique(y_te)) < 2:
                continue

            ev = LinearEval(self.encoder, n_epochs=self.n_epochs_head, lr=self.lr_head, batch_size=self.batch_size)
            ev.fit(X_tr, y_tr)
            res = ev.evaluate(X_te, y_te, dataset_name=dataset_name)
            res.protocol = "LOSO"
            res.extra["held_out_subject"] = int(held_out)
            per_subject.append(res)

        if not per_subject:
            raise ValueError("No valid LOSO folds produced.")

        aurocs = [r.auroc for r in per_subject if np.isfinite(r.auroc)]
        f1s   = [r.f1    for r in per_subject if np.isfinite(r.f1)]
        accs  = [r.accuracy for r in per_subject]
        eces  = [r.ece   for r in per_subject]

        from neuroaugment.evaluation.metrics import cross_site_gap
        gap = cross_site_gap({str(r.extra["held_out_subject"]): r.auroc for r in per_subject})

        summary = EvalResult(
            protocol="LOSO_mean",
            dataset=dataset_name,
            auroc=float(np.nanmean(aurocs)),
            f1=float(np.nanmean(f1s)),
            ece=float(np.nanmean(eces)),
            accuracy=float(np.nanmean(accs)),
            n_train=sum(r.n_train for r in per_subject),
            n_test=sum(r.n_test for r in per_subject),
            extra={
                "auroc_std": float(np.nanstd(aurocs)),
                "f1_std": float(np.nanstd(f1s)),
                "cross_site_gap": gap,
                "n_subjects": len(per_subject),
            },
        )
        return per_subject, summary
