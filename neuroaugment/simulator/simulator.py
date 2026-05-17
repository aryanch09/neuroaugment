from __future__ import annotations

import numpy as np

from neuroaugment.core.causal_model import CausalGenerativeModel, DeviceParams, SiteNoiseParams
from neuroaugment.simulator.state_machine import physiological_events
from neuroaugment.simulator.templates import template_bank


class Simulator:
    def __init__(self, modality: str = "ecg", T: int = 1000, C: int = 3, fs: float = 250.0, seed: int = 0):
        self.modality, self.T, self.C, self.fs = modality, int(T), int(C), float(fs)
        self.rng = np.random.default_rng(seed)

    def _subject(self) -> dict:
        phi = self.rng.normal(0.2, 0.4, size=(self.C, 3))
        if self.modality == "ecg":
            rate = self.rng.normal(1.2, 0.08)
            events = [(o, int(0.12 * self.fs), 0, 1.0) for o in physiological_events(self.T, self.fs, rate, 0.35, self.rng)]
            freqs = np.array([rate, 0.25, 0.08])
        elif self.modality == "eeg":
            events = [(int(self.rng.integers(0, self.T)), int(0.5 * self.fs), 1, 0.8) for _ in range(3)]
            freqs = np.array([10.0, 4.0, 18.0])
        else:
            events = [(o, int(0.2 * self.fs), 0, 0.6) for o in physiological_events(self.T, self.fs, 1.8, 0.25, self.rng)]
            freqs = np.array([1.8, 0.9, 3.6])
        return {"fs": self.fs, "phi": phi, "freqs": freqs, "events": events}

    def _device(self) -> DeviceParams:
        M = np.eye(self.C) + self.rng.normal(0, 0.02, (self.C, self.C))
        np.fill_diagonal(M, 1.0)
        return DeviceParams(
            gain=self.rng.normal(1.0, 0.05, self.C),
            impulse_response=np.array([0.1, 0.8, 0.1]),
            saturation_threshold=3.0,
            channel_dropout_mask=(self.rng.random(self.C) > 0.02).astype(float),
            mixing_matrix=M,
        )

    def _site(self) -> SiteNoiseParams:
        return SiteNoiseParams(
            ar_coeffs=np.array([0.55, -0.12]),
            noise_std=0.03,
            wander_amplitude=0.02,
            wander_freq=float(self.rng.uniform(0.05, 0.4)),
        )

    def sample(self) -> tuple[np.ndarray, np.ndarray, dict]:
        subject = self._subject()
        model = CausalGenerativeModel(k=3, C=self.C, T=self.T, seed=int(self.rng.integers(0, 2**31 - 1)))
        X, y = model.forward(subject, self._device(), self._site())
        meta = {"modality": self.modality, "fs": self.fs, "phi": subject["phi"], "events": subject["events"], "templates": template_bank(self.modality, self.fs)}
        return X, y, meta

    def sample_batch(self, n_samples: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        xs, ys, metas = [], [], []
        for _ in range(n_samples):
            x, y, m = self.sample()
            xs.append(x)
            ys.append(y)
            metas.append(m)
        return np.stack(xs), np.stack(ys), metas
