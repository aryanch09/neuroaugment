# Intended Use and Responsible Use

NeuroAugment is intended for research on physiological time-series representation learning, augmentation robustness, simulation, privacy, and federated training.

Appropriate uses include:

- Benchmarking augmentation policies for EEG, ECG, and IMU datasets.
- Stress-testing models under device, site, and subject shifts.
- Developing self-supervised and federated learning baselines.
- Generating synthetic data for algorithm development when real data cannot be shared.

Out-of-scope uses:

- Clinical diagnosis, triage, treatment recommendations, or patient monitoring.
- Re-identification, inference of sensitive attributes, or reconstruction of private records.
- Claiming synthetic results transfer to a population without validation on representative real data.

Limitations:

- Synthetic physiology is an approximation and may not reproduce all pathologies, sensor artifacts, or demographic variation.
- Differential privacy accounting is provided for research workflows and must be reviewed before regulated deployment.
- Calibration and fairness should be evaluated per site, device, and population.

Users should report model performance with uncertainty, seeds, data provenance, and config files. When working with human data, follow IRB/ethics approval, consent, data minimization, secure storage, and jurisdiction-specific privacy requirements.
