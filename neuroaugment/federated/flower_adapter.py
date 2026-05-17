from __future__ import annotations

import numpy as np
from typing import Optional

from neuroaugment.federated.aggregation import fedavg


class NumpyFederatedServer:
    def __init__(self, initial_state: dict[str, np.ndarray]):
        self.state = {k: v.copy() for k, v in initial_state.items()}
        self.history: list[dict[str, np.ndarray]] = []

    def round(self, client_updates: list[dict[str, np.ndarray]], weights: Optional[list[float]] = None) -> dict[str, np.ndarray]:
        self.state = fedavg(client_updates, weights)
        self.history.append({k: v.copy() for k, v in self.state.items()})
        return self.state


class FlowerAdapter:
    """Optional bridge for Flower. Kept dependency-light for CPU CI."""

    def __init__(self, server: NumpyFederatedServer):
        self.server = server

    def aggregate_fit(self, results: list[tuple[dict[str, np.ndarray], int]]) -> dict[str, np.ndarray]:
        states = [r[0] for r in results]
        weights = [r[1] for r in results]
        return self.server.round(states, weights)
