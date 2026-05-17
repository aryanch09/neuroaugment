import numpy as np

from neuroaugment.core import operators as ops


def test_each_operator_shape(signal_3c, meta_3c, rng):
    x, m = ops.channel_crosstalk(signal_3c, meta_3c, rng)
    assert x.shape == signal_3c.shape and "crosstalk_matrix" in m
    x, m = ops.device_frequency_response(signal_3c, meta_3c, rng)
    assert x.shape == signal_3c.shape
    x, m = ops.colored_noise_addition(signal_3c, meta_3c, rng)
    assert x.shape == signal_3c.shape
    x, m = ops.channel_dropout(signal_3c, meta_3c, rng, dropout_prob=1.0)
    assert x.shape == signal_3c.shape and m["dropout_mask"].shape == signal_3c.shape
    x, m = ops.temporal_causal_masking(signal_3c, meta_3c, rng)
    assert x.shape == signal_3c.shape and m["mask"].shape[0] == signal_3c.shape[0]


def test_event_and_spectral(signal_3c, meta_3c, rng):
    template = np.hanning(32)
    x, m = ops.event_injection(signal_3c, meta_3c, rng, "beat", {"beat": [template]}, -1, 0.1, 1.0, 0.2, 250)
    assert x.shape == signal_3c.shape
    assert "event_injection" in m
    x, m = ops.latent_spectral_modulation(signal_3c, meta_3c, rng, [5, 20])
    assert x.shape == signal_3c.shape


def test_cross_modal(rng):
    xd = {"eeg": np.zeros((100, 2)), "imu": np.zeros((100, 3))}
    out, meta = ops.cross_modal_consistency(xd, {}, rng, ["eeg", "imu"])
    assert out["eeg"].shape == (100, 2)
    assert "cross_modal" in meta
