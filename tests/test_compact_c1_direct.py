import pytest
import torch
import torch.nn.functional as F

from rmr_core.density_activation import compact_c1_activation
from rmr_v3.losses.voronoi_poisson import scale_normalized_voronoi_poisson_loss
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.canonical import get_canonical_model_config


def test_compact_c1_activation_mathematical_properties():
    """Verify that compact_c1 satisfies strict compact support and C1 continuity."""
    # 1. Exact zero support for z <= 0
    z_neg = torch.tensor([-10.0, -5.0, -1.0, -1e-6, 0.0], dtype=torch.float32)
    y_neg = compact_c1_activation(z_neg, tau=0.5)
    assert torch.all(y_neg == 0.0), f"Compact C1 leaked non-zero values on negative logits: {y_neg}"

    # 2. Smooth quadratic ramp for 0 < z <= tau
    z_ramp = torch.tensor([0.1, 0.25, 0.5], dtype=torch.float32, requires_grad=True)
    y_ramp = compact_c1_activation(z_ramp, tau=0.5)
    expected_y = z_ramp**2 / (2.0 * 0.5)
    assert torch.allclose(y_ramp, expected_y), f"Quadratic ramp mismatch: {y_ramp} vs {expected_y}"

    # Verify non-vanishing derivative on positive support
    y_ramp.sum().backward()
    expected_grad = z_ramp / 0.5
    assert torch.allclose(z_ramp.grad, expected_grad), f"Gradient mismatch: {z_ramp.grad} vs {expected_grad}"

    # 3. Linear transition for z > tau
    z_lin = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32, requires_grad=True)
    y_lin = compact_c1_activation(z_lin, tau=0.5)
    expected_lin = z_lin - 0.25
    assert torch.allclose(y_lin, expected_lin), f"Linear ramp mismatch: {y_lin} vs {expected_lin}"


def test_direct_compact_c1_model_forward_backward():
    """Verify that standalone direct model with compact_c1 respects all Track A constraints."""
    cfg = get_canonical_model_config(
        enable_solver=False,
        head_activation="compact_c1",
        dynamic_scale_routing=False,
        density_curvature=False,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Invariant: Track A strictly <= 104,441 parameters
    assert total_params <= 104441, f"Parameter budget exceeded: {total_params} > 104441"

    # Forward pass: direct carrier identity y == y0
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert torch.equal(out.y, out.y0), "Direct model output y must be strictly identical to carrier y0"
    assert out.y.min().item() >= 0.0, f"Density output contains negative values: {out.y.min().item()}"

    # Backward pass with Scale-Normalized Voronoi Poisson loss
    model.train()
    out = model(x)
    points = [torch.tensor([[100.0, 100.0], [120.0, 150.0]]), torch.tensor([[50.0, 50.0]])]
    loss = scale_normalized_voronoi_poisson_loss(out.y, points, stride=4)
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"

    loss.backward()
    has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient detected in parameter {name}"
    assert has_grad, "No gradients were computed during backward pass"
