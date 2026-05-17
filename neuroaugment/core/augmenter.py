from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class Augmenter:
    """Composes augmentation operators with deterministic seeding."""

    def __init__(
        self,
        ops: list[Callable],
        op_probabilities: Optional[list[float]] = None,
        seed: int = 0,
        enforce_cross_channel: bool = True,
        p_apply_any: float = 1.0,
    ) -> None:
        self.ops = list(ops)
        self.op_probabilities = op_probabilities or [1.0] * len(self.ops)
        if len(self.ops) != len(self.op_probabilities):
            raise ValueError("ops and op_probabilities must have the same length")
        self.seed = int(seed)
        self.enforce_cross_channel = bool(enforce_cross_channel)
        self.p_apply_any = float(p_apply_any)

    def _verify(self, before: np.ndarray, after: np.ndarray, meta: dict) -> None:
        if "phi" not in meta or before.shape[1] < 2:
            return
        delta = after - before
        if not np.all(np.isfinite(delta)):
            raise ValueError("Augmentation produced non-finite values")
        phi = np.asarray(meta["phi"])
        shared = np.where(np.any(np.abs(phi) > 1e-8, axis=1))[0] if phi.ndim == 2 else np.arange(before.shape[1])
        if shared.size > 1 and np.allclose(delta[:, shared].std(axis=0), 0) and delta[:, shared].std() > 0:
            raise ValueError("Degenerate cross-channel perturbation detected")

    def _default_params(self, op: Callable, X: np.ndarray, meta: dict) -> dict:
        name = getattr(op, "__name__", "")
        if name == "event_injection":
            t = np.linspace(-1, 1, 64)
            return {
                "event_type": "generic",
                "template_library": {"generic": [np.exp(-8 * t**2) * np.sin(10 * np.pi * t)]},
                "amplitude_lognormal_mu": -1.0,
                "amplitude_lognormal_sigma": 0.2,
                "onset_rate_lambda": 0.5,
                "refractory_period_s": 0.5,
                "fs": meta.get("fs", 250.0),
            }
        if name == "latent_spectral_modulation":
            return {"band_hz": [8.0, 12.0], "q_factor": 2.0, "shift_fraction": 0.1, "duration_fraction": 1.0}
        return {}

    def apply(self, X: np.ndarray, meta: Optional[dict] = None, sample_idx: int = 0) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(self.seed + int(sample_idx))
        X_aug = np.asarray(X, dtype=float).copy()
        meta_out = dict(meta or {})
        meta_out.setdefault("ops_applied", [])
        if rng.random() > self.p_apply_any:
            meta_out["skipped_all"] = True
            return X_aug, meta_out
        for op, prob in zip(self.ops, self.op_probabilities):
            if rng.random() <= prob:
                before = X_aug.copy()
                params = self._default_params(op, X_aug, meta_out)
                X_aug, meta_out = op(X_aug, meta_out, rng, **params)
                if self.enforce_cross_channel:
                    self._verify(before, X_aug, meta_out)
        return X_aug, meta_out

    def apply_pair(self, X: np.ndarray, sample_idx: int = 0) -> tuple[tuple[np.ndarray, dict], tuple[np.ndarray, dict]]:
        device_names = {"device_frequency_response", "channel_crosstalk", "colored_noise_addition", "channel_dropout"}
        dri_ops, dri_probs, all_ops, all_probs = [], [], [], []
        for op, prob in zip(self.ops, self.op_probabilities):
            all_ops.append(op)
            all_probs.append(prob)
            if getattr(op, "__name__", "") in device_names:
                dri_ops.append(op)
                dri_probs.append(prob)
        view1 = Augmenter(dri_ops, dri_probs, self.seed, self.enforce_cross_channel, self.p_apply_any).apply(X, {}, sample_idx)
        view2 = Augmenter(all_ops, all_probs, self.seed + 10_000, self.enforce_cross_channel, self.p_apply_any).apply(
            X, {"previous_view_mask": view1[1].get("mask", np.zeros(X.shape[0], dtype=bool))}, sample_idx
        )
        view1[1]["view_type"] = "DRI"
        view2[1]["view_type"] = "DRI+PLP"
        return view1, view2

    def __repr__(self) -> str:
        names = [getattr(op, "__name__", repr(op)) for op in self.ops]
        return f"Augmenter(ops={names}, seed={self.seed}, p_apply_any={self.p_apply_any})"
