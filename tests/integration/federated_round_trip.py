from __future__ import annotations

import numpy as np

from neuroaugment.federated.aggregation import fedavg


def main() -> None:
    global_state = {"w": np.zeros(2)}
    for _ in range(2):
        clients = [{"w": global_state["w"] + i + 1} for i in range(3)]
        global_state = fedavg(clients, [1, 2, 3])
    assert np.allclose(global_state["w"], np.array([4.66666667, 4.66666667]))
    print(global_state)


if __name__ == "__main__":
    main()
