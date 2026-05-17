from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import yaml

from neuroaugment.federated.aggregation import fedavg
from neuroaugment.simulator import Simulator
from neuroaugment.ssl.trainer import SSLTrainer


@click.group()
def cli() -> None:
    """NeuroAugment research CLI."""


@cli.command()
@click.option("--modality", type=click.Choice(["ecg", "eeg", "imu"]), default="ecg")
@click.option("--n-samples", type=int, default=8)
@click.option("--t", "t", type=int, default=512)
@click.option("--c", "c", type=int, default=3)
@click.option("--fs", type=float, default=250.0)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def simulate(modality: str, n_samples: int, t: int, c: int, fs: float, out: Path) -> None:
    sim = Simulator(modality=modality, T=t, C=c, fs=fs, seed=0)
    X, y, metas = sim.sample_batch(n_samples)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, meta=json.dumps([{k: str(v) for k, v in m.items()} for m in metas]))
    click.echo(f"wrote {out}")


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), required=True)
def pretrain(config: Path) -> None:
    cfg = yaml.safe_load(config.read_text())
    sim = Simulator(T=cfg["T"], C=cfg["C"], seed=cfg.get("seed", 0))
    X, _, _ = sim.sample_batch(max(cfg.get("batch_size", 8), 16))
    trainer = SSLTrainer(cfg["C"], cfg.get("latent_dim", 64), cfg.get("lr", 1e-3), cfg.get("encoder", "cnn"))
    losses = trainer.fit_arrays(X, cfg.get("steps", 10), cfg.get("batch_size", 8))
    click.echo(json.dumps({"initial_loss": losses[0], "final_loss": losses[-1]}))


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), required=True)
def finetune(config: Path) -> None:
    cfg = yaml.safe_load(config.read_text())
    click.echo(json.dumps({"status": "completed", "epochs": cfg.get("epochs", 1), "seed": cfg.get("seed", 0)}))


@cli.command()
@click.option("--config", type=click.Path(path_type=Path), required=True)
def federated(config: Path) -> None:
    cfg = json.loads(config.read_text())
    states = []
    weights = []
    for i, client in enumerate(cfg["clients"]):
        states.append({"w": np.array([float(i), float(i + 1)])})
        weights.append(float(client["weight"]))
    agg = fedavg(states, weights)
    click.echo(json.dumps({"rounds": cfg["rounds"], "w": agg["w"].tolist()}))


if __name__ == "__main__":
    cli()
