import torch
from rmr_v3.losses import bayesian_loss, sinkhorn_ot_loss, RMRv3LossConfig, compute_rmr_v3_losses
from rmr_core.operators import build_multiscale_regions


def test_bayesian_loss_positive_and_gradient():
    torch.manual_seed(42)
    b, h, w = 2, 32, 32
    y0 = torch.rand(b, 1, h, w, requires_grad=True)

    # 2 sample points for img 0, 3 sample points for img 1
    points = [
        torch.tensor([[10.0, 15.0], [20.0, 25.0]]),
        torch.tensor([[5.0, 5.0], [30.0, 30.0], [12.0, 18.0]]),
    ]

    loss = bayesian_loss(y0, points, sigma=8.0, stride=4)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert y0.grad is not None
    assert torch.isfinite(y0.grad).all()


def test_bayesian_loss_empty_points():
    torch.manual_seed(42)
    b, h, w = 2, 16, 16
    y0 = torch.full((b, 1, h, w), 0.1, requires_grad=True)
    points = [torch.empty((0, 2)), None]

    loss = bayesian_loss(y0, points, sigma=8.0, stride=4)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_sinkhorn_ot_loss_positive_and_gradient():
    torch.manual_seed(42)
    b, h, w = 2, 16, 16
    y0 = torch.rand(b, 1, h, w, requires_grad=True)
    points = [
        torch.tensor([[10.0, 15.0]]),
        torch.tensor([[5.0, 5.0], [30.0, 30.0]]),
    ]

    loss = sinkhorn_ot_loss(y0, points, reg=10.0, num_iters=10, stride=4)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert y0.grad is not None
    assert torch.isfinite(y0.grad).all()
