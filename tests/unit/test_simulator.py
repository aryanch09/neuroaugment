from neuroaugment.simulator import Simulator


def test_simulator_shapes_metadata():
    sim = Simulator("ecg", T=300, C=2, seed=5)
    x, y, meta = sim.sample()
    assert x.shape == (300, 2)
    assert y.shape == (300,)
    assert meta["modality"] == "ecg"
    assert "events" in meta
