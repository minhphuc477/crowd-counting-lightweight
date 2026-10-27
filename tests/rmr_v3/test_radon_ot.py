import math
import torch
import pytest

from rmr_v3.losses.radon_ot import radon_sliced_wasserstein_loss


def test_radon_ot_identity_zero():
    """When prediction perfectly matches ground truth, loss must be zero."""
    h, w, stride = 64, 64, 4
    pred = torch.zeros(1, 1, h, w, dtype=torch.float32)
    # Put 1 person at grid cell (10, 15) -> exact pixel center is (15.5 * 4, 10.5 * 4) = (62.0, 42.0)
    pred[0, 0, 10, 15] = 1.0
    pts = [torch.tensor([[62.0, 42.0]], dtype=torch.float32)]
    loss = radon_sliced_wasserstein_loss(pred, pts, stride=stride, num_projections=4)
    assert loss.item() == 0.0, f"Expected 0.0 loss for matching point, got {loss.item()}"


def test_radon_ot_monotonic_shift():
    """Loss must increase monotonically as spatial displacement increases."""
    h, w, stride = 64, 64, 4
    losses = []
    shifts = [0, 2, 5, 10]
    for s in shifts:
        pred = torch.zeros(1, 1, h, w, dtype=torch.float32)
        pred[0, 0, 10 + s, 10 + s] = 1.0
        pts = [torch.tensor([[40.0, 40.0]], dtype=torch.float32)]  # cell (10, 10)
        l = radon_sliced_wasserstein_loss(pred, pts, stride=stride, num_projections=4)
        losses.append(l.item())

    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1], f"Monotonicity violated: {losses}"


def test_radon_ot_zero_points_foreground_penalty():
    """When image contains 0 people, foreground predictions must be heavily penalized."""
    h, w, stride = 32, 32, 4
    pred_clean = torch.zeros(1, 1, h, w, dtype=torch.float32)
    pred_noisy = torch.full((1, 1, h, w), 0.1, dtype=torch.float32)
    pts_empty = [torch.empty((0, 2), dtype=torch.float32)]

    loss_clean = radon_sliced_wasserstein_loss(pred_clean, pts_empty, stride=stride)
    loss_noisy = radon_sliced_wasserstein_loss(pred_noisy, pts_empty, stride=stride)

    assert loss_clean.item() == 0.0
    assert loss_noisy.item() > 0.5


def test_radon_ot_backward_gradients():
    """Verify gradients propagate cleanly to prediction without NaN or Inf."""
    pred = torch.full((2, 1, 32, 32), 0.05, dtype=torch.float32, requires_grad=True)
    pts = [
        torch.tensor([[20.0, 30.0], [50.0, 80.0]], dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
    ]
    loss = radon_sliced_wasserstein_loss(pred, pts, stride=4, num_projections=4)
    loss.backward()

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()
    assert pred.grad.abs().sum() > 0.0


def test_radon_ot_integration_in_allocation_loss():
    """Verify radon_ot and dual_bayesian_radon_ot integrate cleanly into allocation loss routing."""
    from rmr_v3.losses import RMRv3LossConfig, route_allocation_loss

    cfg_rot = RMRv3LossConfig(
        output_stride=4,
        allocation_loss_type="radon_ot",
        lambda_radon_ot=1.0,
        dm_target="y",
    )
    y = torch.full((2, 1, 32, 32), 0.05, dtype=torch.float32, requires_grad=True)
    y0 = torch.full((2, 1, 32, 32), 0.05, dtype=torch.float32, requires_grad=True)
    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    pts = [
        torch.tensor([[20.0, 30.0], [50.0, 80.0]], dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
    ]

    loss_alloc, losses = route_allocation_loss(y, y0, target_y, pts, cfg_rot)
    assert "radon_ot" in losses
    assert "allocation" in losses
    assert loss_alloc.item() > 0.0

    cfg_dual = RMRv3LossConfig(
        output_stride=4,
        allocation_loss_type="dual_bayesian_radon_ot",
        lambda_bayesian=0.15,
        lambda_radon_ot=1.0,
        dm_target="dual",
    )
    loss_dual, losses_dual = route_allocation_loss(y, y0, target_y, pts, cfg_dual)
    assert "radon_ot" in losses_dual
    assert "bayesian" in losses_dual
    assert "allocation" in losses_dual
    assert loss_dual.item() > 0.0

