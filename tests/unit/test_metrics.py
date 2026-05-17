import numpy as np

from neuroaugment.evaluation.metrics import cross_site_gap, ece_score, f1_score_binary


def test_metrics():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.7, 0.4])
    assert f1_score_binary(y, p > 0.5) == 1.0
    assert ece_score(y, p, n_bins=2) >= 0
    assert cross_site_gap({"a": 0.7, "b": 0.9}) == 0.2
