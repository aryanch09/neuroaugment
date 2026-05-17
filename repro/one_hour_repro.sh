#!/usr/bin/env bash
set -euo pipefail
python -m neuroaugment.cli.main simulate --modality ecg --n-samples 8 --out /tmp/neuroaugment_toy.npz
python -m neuroaugment.cli.main pretrain --config configs/pretrain_small.yaml
python -m neuroaugment.cli.main finetune --config configs/finetune.yaml
python paper/figures/plot_metrics.py --out /tmp/neuroaugment_metrics.png
