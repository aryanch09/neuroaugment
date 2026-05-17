from neuroaugment.core.augmenter import Augmenter
from neuroaugment.core.operators import channel_crosstalk, colored_noise_addition


def test_augmenter_reproducible(signal_3c, meta_3c):
    aug = Augmenter([channel_crosstalk, colored_noise_addition], seed=9)
    x1, m1 = aug.apply(signal_3c, meta_3c, sample_idx=4)
    x2, m2 = aug.apply(signal_3c, meta_3c, sample_idx=4)
    assert (x1 == x2).all()
    assert len(m1["ops_applied"]) == len(m2["ops_applied"]) == 2


def test_apply_pair(signal_3c):
    aug = Augmenter([channel_crosstalk, colored_noise_addition], seed=1)
    (x1, m1), (x2, m2) = aug.apply_pair(signal_3c, sample_idx=0)
    assert x1.shape == x2.shape == signal_3c.shape
    assert m1["view_type"] == "DRI"
    assert m2["view_type"] == "DRI+PLP"
