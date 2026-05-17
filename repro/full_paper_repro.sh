#!/usr/bin/env bash
set -euo pipefail
for seed in 1 2 3; do
  python -m neuroaugment.cli.main pretrain --config configs/pretrain_full.yaml
done
python paper/figures/plot_umap.py --out /tmp/neuroaugment_umap.png
python paper/figures/plot_privacy_utility.py --out /tmp/neuroaugment_privacy.png
