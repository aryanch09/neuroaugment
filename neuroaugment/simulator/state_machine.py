from __future__ import annotations

import numpy as np


class HMMStateMachine:
    def __init__(self, transition: np.ndarray, states: list[str], seed: int = 0):
        self.transition = np.asarray(transition, dtype=float)
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.states = states
        self.rng = np.random.default_rng(seed)

    def sample(self, n_steps: int, start_state: int = 0) -> list[str]:
        idx = int(start_state)
        seq = []
        for _ in range(n_steps):
            seq.append(self.states[idx])
            idx = int(self.rng.choice(len(self.states), p=self.transition[idx]))
        return seq


def physiological_events(T: int, fs: float, rate_hz: float, refractory_s: float, rng: np.random.Generator) -> list[int]:
    onsets = []
    t = 0.0
    while t < T:
        t += rng.exponential(fs / max(rate_hz, 1e-6))
        onset = int(round(t))
        if onset >= T:
            break
        if not onsets or onset - onsets[-1] >= refractory_s * fs:
            onsets.append(onset)
    return onsets
