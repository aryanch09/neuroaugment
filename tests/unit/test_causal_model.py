import numpy as np

from neuroaugment.core.causal_model import CausalGenerativeModel, DeviceParams, SiteNoiseParams


def test_causal_forward_labels():
    model = CausalGenerativeModel(2, 2, 100, seed=0)
    subj = {"phi": np.eye(2), "events": [(10, 5, 0, 1.0)], "fs": 250}
    dev = DeviceParams(np.ones(2), np.array([1.0]), 10.0, np.ones(2), np.eye(2))
    site = SiteNoiseParams(np.array([0.5, -0.1]), 0.01, 0.0, 0.1)
    x, y = model.forward(subj, dev, site)
    assert x.shape == (100, 2)
    assert y[10:15].sum() == 5
