from __future__ import annotations

import numpy as np


def ecg_template(fs: float = 250.0) -> np.ndarray:
    t = np.linspace(-0.35, 0.45, int(0.8 * fs))
    p = 0.12 * np.exp(-((t + 0.18) / 0.035) ** 2)
    q = -0.18 * np.exp(-((t + 0.03) / 0.012) ** 2)
    r = 1.00 * np.exp(-(t / 0.014) ** 2)
    s = -0.28 * np.exp(-((t - 0.025) / 0.016) ** 2)
    tw = 0.35 * np.exp(-((t - 0.22) / 0.08) ** 2)
    return p + q + r + s + tw


def eeg_templates(fs: float = 250.0) -> dict[str, np.ndarray]:
    t = np.arange(int(fs)) / fs
    return {
        "alpha_burst": np.sin(2 * np.pi * 10 * t) * np.hanning(len(t)),
        "spindle": np.sin(2 * np.pi * 13 * t) * np.hanning(len(t)),
        "blink": np.exp(-((t - 0.35) / 0.09) ** 2),
    }


def imu_motion_primitive(fs: float = 100.0) -> np.ndarray:
    t = np.arange(int(1.2 * fs)) / fs
    return np.sin(2 * np.pi * 1.8 * t) * np.exp(-2 * np.maximum(0, t - 0.6))


def template_bank(modality: str, fs: float) -> dict[str, list[np.ndarray]]:
    if modality == "ecg":
        base = ecg_template(fs)
        return {"beat": [base, 0.8 * base, np.roll(base, 3)]}
    if modality == "eeg":
        temps = eeg_templates(fs)
        return {k: [v, 0.75 * v, np.roll(v, 5)] for k, v in temps.items()}
    if modality == "imu":
        base = imu_motion_primitive(fs)
        return {"step": [base, 1.2 * base, np.roll(base, 4)]}
    raise ValueError(f"Unsupported modality {modality}")
