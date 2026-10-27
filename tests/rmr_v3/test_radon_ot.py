import math
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.losses.radon_ot import radon_sliced_wasserstein_loss


def test_radon_ot_identity_zero():
    """When prediction perfectly matches ground truth, loss must be near zero (< 1e-6)."""
    h, w, stride = 64, 64, 4
    pred = torch.zeros(1, 1, h, w, dtype=torch.float32)
    pred[0, 0, 10, 15] = 1.0
    pts = [torch.tensor([[62.0, 42.0]], dtype=torch.float32)]  # center of cell (10, 15)
    loss = radon_sliced_wasserstein_loss(pred, pts, stride=stride, num_projections=4)
    assert loss.item() < 1e-6, f"Expected < 1e-6 loss for matching point, got {loss.item()}"


def test_radon_ot_monotonic_shift():
    """Loss must increase monotonically as spatial displacement increases."""
    h, w, stride = 64, 64, 4
    losses = []
    shifts = [0, 2, 5, 10]
    for s in shifts:
        pred = torch.zeros(1, 1, h, w, dtype=torch.float32)
        pred[0, 0, 10 + s, 10 + s] = 1.0
        pts = [torch.tensor([[42.0, 42.0]], dtype=torch.float32)]  # cell (10, 10) center
        l = radon_sliced_wasserstein_loss(pred, pts, stride=stride, num_projections=4)
        losses.append(l.item())

    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1], f"Monotonicity violated: {losses}"


def test_radon_ot_translation_invariance():
    """Shift penalty must be invariant to where the points are located in the image."""
    h, w, stride = 128, 128, 4
    # Shift of 4 cells on the left: target at (16, 16), pred at (16, 20)
    p_left = torch.zeros(1, 1, h, w)
    p_left[0, 0, 16, 20] = 1.0
    pts_left = [torch.tensor([[16.5 * stride, 16.5 * stride]])]
    loss_left = radon_sliced_wasserstein_loss(p_left, pts_left, stride=stride, num_projections=4)

    # Shift of 4 cells on the right: target at (112, 112), pred at (112, 116)
    p_right = torch.zeros(1, 1, h, w)
    p_right[0, 0, 112, 116] = 1.0
    pts_right = [torch.tensor([[112.5 * stride, 112.5 * stride]])]
    loss_right = radon_sliced_wasserstein_loss(p_right, pts_right, stride=stride, num_projections=4)

    ratio = loss_left.item() / max(loss_right.item(), 1e-8)
    assert abs(ratio - 1.0) < 1e-4, f"Translation invariance violated: ratio={ratio}"


def test_radon_ot_gradient_dipole():
    """Gradient at target must be attractive (< 0) and at background must be repulsive (> 0)."""
    h, w, stride = 64, 64, 4
    pred = torch.full((1, 1, h, w), 0.01, dtype=torch.float32, requires_grad=True)
    pts = [torch.tensor([[16.5 * stride, 16.5 * stride]])]  # cell (16, 16)
    loss = radon_sliced_wasserstein_loss(pred, pts, stride=stride, num_projections=4)
    loss.backward()

    grad = pred.grad[0, 0]
    grad_target = grad[16, 16].item()
    grad_background = grad[60, 60].item()

    assert grad_target < 0.0, f"Expected attractive negative gradient at target, got {grad_target}"
    assert grad_background > 0.0, f"Expected repulsive positive gradient at background, got {grad_background}"


def test_radon_ot_zero_points_foreground_penalty():
    """When image contains 0 people, foreground predictions must be penalized."""
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
