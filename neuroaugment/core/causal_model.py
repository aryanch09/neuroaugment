from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class DeviceParams:
    gain: np.ndarray
    impulse_response: np.ndarray
    saturation_threshold: float
    channel_dropout_mask: np.ndarray
    mixing_matrix: np.ndarray


@dataclass
class SiteNoiseParams:
    ar_coeffs: np.ndarray
    noise_std: float
    wander_amplitude: float
    wander_freq: float


class CausalGenerativeModel:
    """Implements X_{s,d,u}(t,c) = D_{s,d}[G_c(P_u(t); phi_c)] + N_{s,d,u}(t,c)."""

    def __init__(self, k: int, C: int, T: int, fs: float = 250.0, seed: int = 0):
        self.k = int(k)
        self.C = int(C)
        self.T = int(T)
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

    def sample_physiological_latent(self, subject_params: dict, n_steps: int) -> np.ndarray:
        fs = float(subject_params.get("fs", 250.0))
        drift = np.asarray(subject_params.get("drift", np.zeros(self.k)), dtype=float)
        freqs = np.asarray(subject_params.get("freqs", np.linspace(0.7, 12.0, self.k)), dtype=float)
        amp = np.asarray(subject_params.get("amplitudes", np.ones(self.k)), dtype=float)
        phases = self.rng.uniform(0, 2 * np.pi, size=self.k)
        t = np.arange(n_steps) / fs
        P = np.zeros((n_steps, self.k), dtype=float)
        for i in range(self.k):
            ar = signal.lfilter([1.0], [1.0, -0.98], self.rng.normal(0, 0.02, n_steps))
            P[:, i] = amp[i % len(amp)] * np.sin(2 * np.pi * freqs[i % len(freqs)] * t + phases[i]) + ar
        P += drift[: self.k]
        events = subject_params.get("events")
        if events:
            for onset, duration, driver, magnitude in events:
                lo, hi = max(0, int(onset)), min(n_steps, int(onset + duration))
                if lo < hi:
                    P[lo:hi, int(driver) % self.k] += float(magnitude)
        return P

    def apply_channel_mapping(self, P: np.ndarray, phi: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if P.ndim != 2 or phi.shape != (self.C, self.k):
            raise ValueError(f"Expected P (T,{self.k}) and phi ({self.C},{self.k})")
        X = P @ phi.T
        fs = self.fs
        nyq = fs / 2 - 1.0
        bands = [(0.5, min(40.0, nyq)), (0.5, min(30.0, nyq)), (1.0, min(45.0, nyq)), (0.1, min(20.0, nyq))]
        for c in range(self.C):
            low, high = bands[c % len(bands)]
            sos = signal.butter(2, [low, min(high, fs / 2 - 1)], btype="bandpass", fs=fs, output="sos")
            if X.shape[0] > 27:
                X[:, c] = signal.sosfiltfilt(sos, X[:, c])
            else:
                X[:, c] = signal.sosfilt(sos, X[:, c])
        return X

    def apply_device_transform(self, signal_in: np.ndarray, device_params: DeviceParams) -> np.ndarray:
        X = np.asarray(signal_in, dtype=float).copy()
        if X.shape[1] != self.C:
            raise ValueError(f"Expected {self.C} channels")
        X *= np.asarray(device_params.gain)[None, :]
        fir = np.asarray(device_params.impulse_response, dtype=float)
        for c in range(self.C):
            X[:, c] = signal.fftconvolve(X[:, c], fir, mode="same")
        X = X @ np.asarray(device_params.mixing_matrix, dtype=float).T
        X = np.tanh(X / device_params.saturation_threshold) * device_params.saturation_threshold
        X *= np.asarray(device_params.channel_dropout_mask, dtype=float)[None, :]
        return X

    def sample_noise(self, T: int, C: int, site_params: SiteNoiseParams) -> np.ndarray:
        ar = np.asarray(site_params.ar_coeffs, dtype=float)
        noise = np.zeros((T, C), dtype=float)
        for c in range(C):
            eps = self.rng.normal(0, site_params.noise_std, size=T)
            noise[:, c] = signal.lfilter([1.0], np.r_[1.0, -ar[:2]], eps)
        t = np.arange(T) / self.fs
        phase = self.rng.uniform(0, 2 * np.pi, C)
        wander = site_params.wander_amplitude * np.sin(
            2 * np.pi * site_params.wander_freq * t[:, None] + phase[None, :]
        )
        return noise + wander

    def forward(
        self,
        subject_params: dict,
        device_params: DeviceParams,
        site_params: SiteNoiseParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        phi = np.asarray(subject_params.get("phi", self.rng.normal(0, 1, (self.C, self.k))))
        P = self.sample_physiological_latent(subject_params, self.T)
        mapped = self.apply_channel_mapping(P, phi)
        X = self.apply_device_transform(mapped, device_params)
        X = X + self.sample_noise(self.T, self.C, site_params)
        y = np.zeros(self.T, dtype=int)
        for onset, duration, *_ in subject_params.get("events", []):
            y[max(0, int(onset)) : min(self.T, int(onset + duration))] = 1
        return X.astype(np.float32), y


class CausalModelFitter:
    """Fits device and site priors from calibration data."""

    def __init__(self) -> None:
        self.device_prior: dict[str, Any] = {}
        self.site_prior: dict[str, Any] = {}

    def _as_array(self, calibration_data: Any) -> np.ndarray:
        if hasattr(calibration_data, "signals"):
            X = calibration_data.signals
        else:
            X = calibration_data
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3:
            raise ValueError("Calibration data must be (N,T,C) or (T,C)")
        return X

    def fit_device_prior(self, calibration_data: Any) -> dict:
        X = self._as_array(calibration_data)
        gains = X.std(axis=(0, 1))
        gains = gains / (np.mean(gains) + 1e-12)
        fs = 250.0
        psds = []
        for c in range(X.shape[2]):
            _, pxx = signal.welch(X[:, :, c].reshape(-1), fs=fs, nperseg=min(256, X.shape[1]))
            psds.append(pxx)
        mean_psd = np.mean(psds, axis=0)
        filt_len = 17
        impulse = signal.firwin(filt_len, cutoff=40.0, fs=fs)
        self.device_prior = {
            "gain_mean": gains,
            "gain_std": np.full(X.shape[2], 0.05),
            "impulse_response": impulse,
            "saturation_threshold": float(np.percentile(np.abs(X), 99.5) + 1e-6),
            "mean_psd": mean_psd,
        }
        resid = np.diff(X, axis=1)
        self.site_prior = {
            "noise_std": float(np.std(resid) / np.sqrt(2)),
            "ar_mean": np.array([0.6, -0.15]),
            "ar_std": np.array([0.05, 0.03]),
            "wander_amplitude": float(np.percentile(np.abs(X.mean(axis=2)), 90) * 0.05 + 1e-4),
        }
        return {"device": self.device_prior, "site": self.site_prior}

    def sample_device(self, rng: np.random.Generator) -> DeviceParams:
        if not self.device_prior:
            self.device_prior = {
                "gain_mean": np.ones(3),
                "gain_std": np.full(3, 0.05),
                "impulse_response": signal.firwin(17, 40, fs=250),
                "saturation_threshold": 3.0,
            }
        C = len(self.device_prior["gain_mean"])
        M = np.eye(C) + rng.normal(0, 0.02, (C, C))
        np.fill_diagonal(M, 1.0)
        return DeviceParams(
            gain=rng.normal(self.device_prior["gain_mean"], self.device_prior["gain_std"]),
            impulse_response=np.asarray(self.device_prior["impulse_response"]),
            saturation_threshold=float(self.device_prior["saturation_threshold"]),
            channel_dropout_mask=(rng.random(C) > 0.02).astype(float),
            mixing_matrix=M,
        )

    def sample_site_noise(self, rng: np.random.Generator) -> SiteNoiseParams:
        if not self.site_prior:
            self.site_prior = {
                "noise_std": 0.03,
                "ar_mean": np.array([0.6, -0.15]),
                "ar_std": np.array([0.05, 0.03]),
                "wander_amplitude": 0.02,
            }
        ar = rng.normal(self.site_prior["ar_mean"], self.site_prior["ar_std"])
        if np.sum(np.abs(ar)) >= 0.98:
            ar = ar / (np.sum(np.abs(ar)) + 1e-12) * 0.9
        return SiteNoiseParams(
            ar_coeffs=ar,
            noise_std=float(max(1e-6, rng.normal(self.site_prior["noise_std"], 0.005))),
            wander_amplitude=float(max(0, rng.normal(self.site_prior["wander_amplitude"], 0.005))),
            wander_freq=float(rng.uniform(0.05, 0.5)),
        )
