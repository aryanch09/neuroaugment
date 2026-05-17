# NeuroAugment

NeuroAugment is a causality-aware augmentation and simulation suite for multimodal physiological time series, focused on EEG, ECG, and IMU research workflows. It provides deterministic augmentation operators, a formal causal generative model, synthetic simulators, SSL training utilities, federated learning primitives, privacy accounting, evaluation metrics, and reproducible command-line experiments.

## Highlights

- Causal signal model: `X_{s,d,u}(t,c) = D_{s,d}[G_c(P_u(t); phi_c)] + N_{s,d,u}(t,c)`.
- Cross-channel consistent physiological interventions with independent device/noise shifts.
- ECG, EEG, and IMU simulators with label traces and metadata.
- Contrastive SSL components: temporal CNN/Transformer encoders, projection heads, InfoNCE, invariance, and causal consistency losses.
- Federated aggregation and communication compression utilities.
- Differential privacy helpers and RDP epsilon accounting.
- Reproducible CLI, configs, notebooks, tests, and Docker images.

## Install

```bash
cd neuroaugment
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

For a conda environment:

```bash
conda env create -f environment.yml
conda activate neuroaugment
pip install -e ".[dev]"
```

## Quickstart

```bash
neuroaugment simulate --modality ecg --n-samples 8 --out /tmp/neuroaugment_sim.npz
neuroaugment pretrain --config configs/pretrain_small.yaml
neuroaugment finetune --config configs/finetune.yaml
```

Python:

```python
import numpy as np
from neuroaugment.core import Augmenter
from neuroaugment.core.operators import colored_noise_addition, channel_crosstalk

X = np.random.randn(500, 3)
aug = Augmenter([channel_crosstalk, colored_noise_addition], seed=7)
X_aug, meta = aug.apply(X, sample_idx=0)
```

## Reproducibility

All stochastic components accept explicit seeds. The augmenter seeds each sample as `base_seed + sample_idx`, making repeated runs bitwise deterministic for NumPy operations. Run:

```bash
make test
./repro/one_hour_repro.sh
```

## Scope

NeuroAugment is a research tool. It does not diagnose, treat, or monitor disease and should not be deployed in clinical workflows without validation, governance, and approval. See `INTENDED_USE.md`.

## Citation

If you use NeuroAugment in research, cite the repository and include the exact git commit, config files, and random seeds.
