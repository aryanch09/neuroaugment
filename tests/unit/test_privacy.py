from neuroaugment.privacy.accounting import rdp_epsilon


def test_epsilon_monotone_steps():
    e1 = rdp_epsilon(0.01, 1.0, 10)
    e2 = rdp_epsilon(0.01, 1.0, 20)
    assert e2 >= e1 > 0
