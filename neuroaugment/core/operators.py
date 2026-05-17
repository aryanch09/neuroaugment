from __future__ import annotations

import numpy as np
from scipy import interpolate, signal
from typing import Optional, Sequence, Tuple

from neuroaugment.core.registry import register


def _copy_meta(meta: Optional[dict]) -> dict:
    out = dict(meta or {})
    out.setdefault("ops_applied", [])
    return out


def _record(meta: dict, name: str, params: dict) -> None:
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.copy()
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        else:
            clean[k] = v
    meta.setdefault("ops_applied", []).append({"name": name, "params": clean})


def _stationary_ar2(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        a = rng.uniform(-0.8, 0.8, size=2)
        roots = np.roots([1.0, -a[0], -a[1]])
        if np.all(np.abs(roots) < 1):
            return a
    return np.array([0.5, -0.1])


@register()
def event_injection(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    event_type: str,
    template_library: dict,
    amplitude_lognormal_mu: float,
    amplitude_lognormal_sigma: float,
    onset_rate_lambda: float,
    refractory_period_s: float,
    fs: float,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    T, C = X.shape
    phi = np.asarray(meta.get("phi", np.eye(C)), dtype=float)
    templates = template_library[event_type]
    onsets, amplitudes, template_ids = [], [], []
    t = 0.0
    refractory = refractory_period_s * fs
    while t < T:
        t += rng.exponential(fs / max(onset_rate_lambda, 1e-12))
        onset = int(round(t))
        if onsets and onset - onsets[-1] < refractory:
            continue
        if onset >= T:
            break
        idx = int(rng.integers(0, len(templates)))
        template = np.asarray(templates[idx], dtype=float).reshape(-1)
        warp = float(rng.uniform(0.85, 1.15))
        new_len = max(3, int(round(len(template) * warp)))
        f = interpolate.interp1d(np.arange(len(template)), template, kind="cubic")
        warped = f(np.linspace(0, len(template) - 1, new_len))
        amp = float(rng.lognormal(amplitude_lognormal_mu, amplitude_lognormal_sigma))
        latent = amp * warped
        if phi.ndim == 2:
            projection = phi[:, 0] if phi.shape[1] > 0 else np.ones(C)
            projection = projection[:C]
        else:
            projection = np.ones(C)
        event = latent[:, None] * projection[None, :]
        end = min(T, onset + new_len)
        X[onset:end, :] += event[: end - onset]
        onsets.append(onset)
        amplitudes.append(amp)
        template_ids.append(idx)
    meta.setdefault("event_injection", []).append(
        {"event_type": event_type, "onsets": onsets, "amplitudes": amplitudes, "template_ids": template_ids}
    )
    _record(meta, "event_injection", {"event_type": event_type, "onsets": onsets, "amplitudes": amplitudes})
    return X, meta


@register()
def latent_spectral_modulation(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    band_hz: Sequence[float],
    q_factor: float = 2.0,
    shift_fraction: float = 0.1,
    duration_fraction: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Modulate energy within a physiological frequency band.

    Args:
        band_hz: [f_low, f_high] in Hz defining the target band.
        q_factor: Controls selectivity — higher Q concentrates the gain boost
            near the band centre (narrower Gaussian taper); lower Q spreads it
            uniformly across the full band.  Must be > 0.
        shift_fraction: Fractional gain applied at the band centre (e.g. 0.1 = +10%).
        duration_fraction: Fraction of STFT columns (time) to modulate.
    """
    X = np.asarray(X, dtype=float)
    meta = _copy_meta(meta)
    T, C = X.shape
    fs = float(meta.get("fs", 250.0))
    nperseg = min(256, max(16, T))

    # Draw a random centre frequency within the band (uses rng for reproducibility)
    f_center = float(rng.uniform(band_hz[0], band_hz[1]))
    # q_factor: bandwidth = band_width / q_factor; higher Q → narrower taper
    bandwidth = max((band_hz[1] - band_hz[0]) / max(q_factor, 1e-2), 1e-3)

    out = np.zeros_like(X)
    for c in range(C):
        f, _tt, Z = signal.stft(X[:, c], fs=fs, window="hann", nperseg=nperseg)
        # Gaussian taper centred at f_center with width controlled by q_factor
        taper = np.exp(-0.5 * ((f - f_center) / (bandwidth / 2 + 1e-9)) ** 2)
        dur_cols = max(1, int(np.ceil(Z.shape[1] * duration_fraction)))
        Z[:, :dur_cols] *= 1.0 + shift_fraction * taper[:, None]
        _, rec = signal.istft(Z, fs=fs, window="hann", nperseg=nperseg)
        out[:, c] = rec[:T] if len(rec) >= T else np.pad(rec, (0, T - len(rec)))
    meta["spectral_modulation"] = {
        "band_hz": list(band_hz),
        "f_center": f_center,
        "q_factor": q_factor,
        "shift_fraction": float(shift_fraction),
    }
    _record(meta, "latent_spectral_modulation", meta["spectral_modulation"])
    return out, meta


@register()
def device_frequency_response(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    fc_log_uniform_low: float = 0.1,
    fc_log_uniform_high: float = 40.0,
    filter_order: int = 2,
    fs: float = 250.0,
    apply_zero_phase: bool = True,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    T, C = X.shape
    fc = float(np.exp(rng.uniform(np.log(fc_log_uniform_low), np.log(fc_log_uniform_high))))
    ftype = "lowpass" if fc < 0.8 else str(rng.choice(["bandpass", "lowpass"]))
    if ftype == "bandpass":
        wn = [0.5, min(fc, fs / 2 - 1e-3)]
        if wn[1] <= wn[0]:
            ftype, wn = "lowpass", min(max(fc, 0.05), fs / 2 - 1e-3)
    else:
        wn = min(max(fc, 0.05), fs / 2 - 1e-3)
    sos = signal.butter(filter_order, wn, btype=ftype, fs=fs, output="sos")
    for c in range(C):
        if apply_zero_phase and T > 3 * (2 * sos.shape[0] + 1):
            X[:, c] = signal.sosfiltfilt(sos, X[:, c])
        else:
            X[:, c] = signal.sosfilt(sos, X[:, c])
    gains = rng.normal(1.0, 0.05, size=C)
    X *= gains[None, :]
    meta["device_frequency_response"] = {"fc": fc, "filter_type": ftype, "gains": gains}
    _record(meta, "device_frequency_response", meta["device_frequency_response"])
    return X, meta


@register()
def channel_crosstalk(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    off_diagonal_scale: float = 0.05,
    sparsity: float = 0.7,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    C = X.shape[1]
    M = np.eye(C)
    off = rng.normal(0, off_diagonal_scale, size=(C, C))
    mask = rng.random((C, C)) > sparsity
    off *= mask
    np.fill_diagonal(off, 0.0)
    M += off
    Y = (M @ X.T).T
    meta["crosstalk_matrix"] = M
    _record(meta, "channel_crosstalk", {"M": M})
    return Y, meta


@register()
def colored_noise_addition(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    ar_order: int = 2,
    noise_std_range: Tuple[float, float] = (0.01, 0.1),
    snr_db_range: Tuple[float, float] = (15, 40),
    fs: float = 250.0,
) -> tuple[np.ndarray, dict]:
    if ar_order != 2:
        raise ValueError("colored_noise_addition currently implements stationary AR(2)")
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    T, C = X.shape
    coeffs = _stationary_ar2(rng)
    target_snr = float(rng.uniform(*snr_db_range))
    noise = np.zeros_like(X)
    for c in range(C):
        std = float(rng.uniform(*noise_std_range))
        raw = signal.lfilter([1.0], np.r_[1.0, -coeffs], rng.normal(0, std, size=T))
        sig_power = np.mean(X[:, c] ** 2) + 1e-12
        noise_power = np.mean(raw**2) + 1e-12
        scale = np.sqrt(sig_power / (10 ** (target_snr / 10)) / noise_power)
        noise[:, c] = raw * scale
    t = np.arange(T) / fs
    A = float(rng.uniform(0, 0.05))
    fw = float(rng.uniform(0.05, 0.5))
    phase = rng.uniform(0, 2 * np.pi, size=C)
    wander = A * np.sin(2 * np.pi * fw * t[:, None] + phase[None, :])
    X = X + noise + wander
    meta["colored_noise"] = {"ar_coeffs": coeffs, "snr_db": target_snr, "wander_amplitude": A, "wander_freq": fw}
    _record(meta, "colored_noise_addition", meta["colored_noise"])
    return X, meta


@register()
def channel_dropout(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    dropout_prob: float = 0.1,
    max_consecutive_samples: int = 50,
    mode: str = "zero",
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    T, C = X.shape
    mask = np.zeros((T, C), dtype=bool)
    ranges: list[dict] = []
    for c in range(C):
        n_blocks = rng.poisson(dropout_prob * max(1, T / max_consecutive_samples))
        for _ in range(n_blocks):
            start = int(rng.integers(0, T))
            length = int(rng.integers(1, max_consecutive_samples + 1))
            end = min(T, start + length)
            mask[start:end, c] = True
            ranges.append({"channel": c, "start": start, "end": end})
            if mode == "zero":
                X[start:end, c] = 0.0
            elif mode == "interpolate":
                left, right = max(0, start - 1), min(T - 1, end)
                X[start:end, c] = np.linspace(X[left, c], X[right, c], end - start + 2)[1:-1]
            else:
                raise ValueError("mode must be 'zero' or 'interpolate'")
    meta["dropout_mask"] = mask
    meta["dropout_ranges"] = ranges
    _record(meta, "channel_dropout", {"ranges": ranges, "mode": mode})
    return X, meta


@register()
def temporal_causal_masking(
    X: np.ndarray,
    meta: dict,
    rng: np.random.Generator,
    mask_ratio: float = 0.15,
    mask_len_range: Tuple[int, int] = (10, 50),
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float).copy()
    meta = _copy_meta(meta)
    T = X.shape[0]
    mask = np.zeros(T, dtype=bool)
    target = int(round(mask_ratio * T))
    attempts = 0
    previous = np.asarray(meta.get("previous_view_mask", np.zeros(T, dtype=bool)), dtype=bool)
    while mask.sum() < target and attempts < 10 * T:
        length = int(rng.integers(mask_len_range[0], mask_len_range[1] + 1))
        start = int(rng.integers(0, max(1, T - length + 1)))
        proposal = np.zeros(T, dtype=bool)
        proposal[start : start + length] = True
        proposal &= ~previous
        mask |= proposal
        attempts += 1
    X[mask, :] = 0.0
    meta["mask"] = mask
    _record(meta, "temporal_causal_masking", {"mask_ratio": float(mask.mean()), "mask_len_range": mask_len_range})
    return X, meta


@register()
def cross_modal_consistency(
    X_dict: dict[str, np.ndarray],
    meta: dict,
    rng: np.random.Generator,
    modalities: list[str],
    correlation_strength: float = 0.7,
    time_lag_samples: int = 5,
) -> tuple[dict[str, np.ndarray], dict]:
    meta = _copy_meta(meta)
    T = min(np.asarray(X_dict[m]).shape[0] for m in modalities)
    mode = str(rng.choice(["step", "sinusoid"]))
    if mode == "step":
        shift = np.zeros(T)
        shift[int(T * 0.35) : int(T * 0.7)] = rng.normal(1.0, 0.1)
    else:
        shift = np.sin(2 * np.pi * rng.uniform(0.2, 2.0) * np.linspace(0, 1, T))
    out = {k: np.asarray(v, dtype=float).copy() for k, v in X_dict.items()}
    corr = {}
    for i, m in enumerate(modalities):
        X = out[m]
        lag = i * time_lag_samples
        s = np.roll(shift, lag)
        weight = correlation_strength * (1.0 - 0.1 * i)
        X[:T, :] += weight * s[:, None]
        out[m] = X
        corr[m] = {"lag": lag, "weight": weight}
    meta["cross_modal"] = {"mode": mode, "correlation": corr}
    _record(meta, "cross_modal_consistency", meta["cross_modal"])
    return out, meta
