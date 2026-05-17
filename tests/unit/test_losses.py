import torch

from neuroaugment.ssl.losses import causal_consistency_loss, info_nce_loss, invariance_loss


def test_info_nce_gradient():
    z1 = torch.randn(4, 8, requires_grad=True)
    z2 = torch.randn(4, 8, requires_grad=True)
    loss = info_nce_loss(z1, z2)
    loss.backward()
    assert z1.grad is not None
    assert torch.isfinite(loss)


def test_invariance_and_causal_loss():
    x = torch.randn(3, 5)
    assert invariance_loss(x, x).item() < 1e-6
    assert causal_consistency_loss(x, x).item() < 1e-6
