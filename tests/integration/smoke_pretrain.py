from __future__ import annotations

from neuroaugment.simulator import Simulator
from neuroaugment.ssl.trainer import SSLTrainer


def main() -> None:
    sim = Simulator(T=128, C=3, seed=0)
    X, _, _ = sim.sample_batch(12)
    trainer = SSLTrainer(3, latent_dim=16, lr=1e-3)
    losses = trainer.fit_arrays(X, steps=10, batch_size=4)
    assert len(losses) == 10
    assert losses[-1] <= max(losses[:3]) + 1e-4
    print({"initial": losses[0], "final": losses[-1]})


if __name__ == "__main__":
    main()
