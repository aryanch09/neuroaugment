from __future__ import annotations

import numpy as np
from typing import Optional


State = dict[str, np.ndarray]


def _zeros_like(state: State) -> State:
    return {k: np.zeros_like(v) for k, v in state.items()}


def fedavg(client_states: list[State], weights: Optional[list[float]] = None) -> State:
    """Federated Averaging (McMahan et al., 2017).

    Weighted mean of client model states.  Weights default to uniform if omitted
    (pass dataset sizes for proper data-proportional aggregation).
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")
    w = np.ones(len(client_states)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    out = _zeros_like(client_states[0])
    for state, wi in zip(client_states, w):
        for k in out:
            out[k] = out[k] + wi * state[k]
    return out


def fedprox_aggregate(
    client_states: list[State],
    global_state: State,
    mu: float = 0.01,
    weights: Optional[list[float]] = None,
) -> State:
    """Server-side proximal shrinkage after FedAvg aggregation.

    Applies a soft pull toward the previous global model:
        w_new = fedavg(clients) * (1 - mu) + global * mu

    Note: the canonical FedProx algorithm (Li et al., 2020) adds the proximal
    term ``mu/2 ||w - w_global||^2`` to each *client's local objective* during
    training.  This function is a server-side approximation useful when clients
    cannot be modified; for full FedProx, apply the proximal regulariser locally
    on each client before calling fedavg().
    """
    avg = fedavg(client_states, weights)
    return {k: (1.0 - mu) * avg[k] + mu * global_state[k] for k in avg}


# Keep legacy name for backwards compatibility
def fedprox(
    client_states: list[State],
    global_state: State,
    mu: float = 0.01,
    weights: Optional[list[float]] = None,
) -> State:
    """Alias for fedprox_aggregate; prefer that name for clarity."""
    return fedprox_aggregate(client_states, global_state, mu, weights)


def scaffold(
    client_states: list[State],
    control_variates: list[State],
    server_control: State,
    weights: Optional[list[float]] = None,
) -> tuple[State, State]:
    """SCAFFOLD aggregation (Karimireddy et al., 2020).

    Corrects client drift by subtracting client control variates and adding
    the server control variate before averaging.  Returns the new global state
    and updated server control variate.
    """
    corrected = [
        {k: state[k] - cv[k] + server_control[k] for k in state}
        for state, cv in zip(client_states, control_variates)
    ]
    new_state = fedavg(corrected, weights)
    # Server control update: average of client control variates
    new_control = fedavg(control_variates, weights)
    return new_state, new_control
