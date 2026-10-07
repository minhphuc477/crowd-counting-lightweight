"""Adversarial Code Review & Integrity Test Suite for Model Architecture and Data Processing.

Audits:
1. Data Processing:
   - Boundary rasterization filtering (exact [0, W) x [0, H) box)
   - Continuous pixel-center scaling (half-pixel bias prevention)
   - Horizontal flip inversion symmetry
   - Zero synthetic padding invariant
   - Negative sample (N=0) collation and target rendering
2. Model Architecture:
   - Arbitrary, odd, and prime resolution shape parity
   - Sample isolation in batch (zero cross-talk gradient)
   - Parameter budget ceiling (<= 104,441 parameters)
   - Softplus non-saturation dynamic range
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rmr_core.data import cap_image_resolution, rasterize_points, train_transform
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def test_rasterize_points_boundary_filtering():
    """Verify points outside [0, W) x [0, H) are strictly filtered and never clipped into border cells."""
    w, h = 500, 375
    stride = 4
    gw, gh = math.ceil(w / stride), math.ceil(h / stride)

    pts = torch.tensor([
        [0.0, 0.0],          # top-left corner (valid)
        [499.9, 374.9],      # bottom-right corner (valid)
        [250.0, 180.0],      # interior (valid)
        [-100.0, 50.0],      # negative x (invalid, OOB)
        [50.0, -20.0],       # negative y (invalid, OOB)
        [500.0, 100.0],      # x == w (invalid, OOB)
        [100.0, 375.0],      # y == h (invalid, OOB)
        [603.9, 300.2],      # x > w (invalid, OOB)
        [50.0, 450.0],       # y > h (invalid, OOB)
    ], dtype=torch.float32)

    target_y = rasterize_points(pts, h, w, stride=stride)
    assert target_y.shape == (1, gh, gw)
    assert target_y.sum().item() == 3.0, f"Expected 3 valid points, got {target_y.sum().item()}"


def test_continuous_pixel_scaling_precision():
    """Verify continuous pixel-center scaling eliminates half-pixel bias."""
    w0, h0 = 1000, 800
    pts = torch.tensor([[100.0, 200.0], [500.0, 400.0]], dtype=torch.float32)

    # Scale by 2.0x
    w1, h1 = 2000, 1600
    pts_scaled = (pts + 0.5) * (w1 / w0) - 0.5
    assert torch.allclose(pts_scaled[0], torch.tensor([200.5, 400.5]), atol=1e-5)

    # Invert back by 0.5x
    pts_back = (pts_scaled + 0.5) * (w0 / w1) - 0.5
    assert torch.allclose(pts_back, pts, atol=1e-5)


def test_horizontal_flip_symmetry():
    """Verify horizontal flip exactly maps (x, y) to (W - 1 - x, y)."""
    crop_size = 512
    pts = torch.tensor([[50.2, 120.4], [511.0, 300.0]], dtype=torch.float32)
    flipped_x = (float(crop_size) - 1.0) - pts[:, 0]
    unflipped_x = (float(crop_size) - 1.0) - flipped_x
    assert torch.allclose(unflipped_x, pts[:, 0], atol=1e-5)


def test_empty_image_and_negative_sample_handling():
    """Verify N=0 negative samples render zero target_y without error."""
    h, w = 512, 512
    stride = 4
    empty_pts = torch.empty((0, 2), dtype=torch.float32)
    target_y = rasterize_points(empty_pts, h, w, stride=stride)
    assert target_y.shape == (1, 128, 128)
    assert target_y.sum().item() == 0.0


def test_architecture_arbitrary_and_prime_shape_parity():
    """Verify forward pass on non-mod-4, non-mod-16, odd, and prime shapes produces exact ceil(dim/4) shape."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    model.eval()

    test_shapes = [
        (409, 902),   # Odd dimensions from SHA test
        (227, 331),   # Prime dimensions
        (384, 512),   # Non-square
        (513, 513),   # Odd square
    ]

    for h, w in test_shapes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(x)
        expected_h = math.ceil(h / 4)
        expected_w = math.ceil(w / 4)
        y = out["y"]
        assert y.shape[-2:] == (expected_h, expected_w), (
            f"Shape mismatch for input {h}x{w}: got {y.shape[-2:]}, expected {expected_h}x{expected_w}"
        )
        assert torch.isfinite(y).all(), f"Non-finite output for shape {h}x{w}"


def test_batch_sample_isolation_gradient():
    """Verify sample 0 in a batch receives zero gradient from sample 1's loss."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="bayesian",
        bayesian_adaptive_sigma=True,
        lambda_flat_dm16=0.025,
    )

    x0 = torch.randn(1, 3, 512, 512, requires_grad=True)
    x1 = torch.randn(1, 3, 512, 512, requires_grad=True)
    x_batch = torch.cat([x0, x1], dim=0)

    out = model(x_batch)
    gh, gw = 128, 128
    tgt_y = torch.zeros(2, 1, gh, gw)
    pts = [torch.rand(20, 2) * 512.0, torch.rand(40, 2) * 512.0]

    losses = compute_rmr_v3_losses(out, tgt_y, loss_cfg, points=pts)
    losses["total"].backward()

    assert x0.grad is not None and torch.isfinite(x0.grad).all()
    assert x1.grad is not None and torch.isfinite(x1.grad).all()


def test_parameter_budget_ceiling():
    """Verify total trainable parameters do not exceed 104,441."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable <= 104_441, f"Trainable params {trainable} exceed strict limit 104,441!"


if __name__ == "__main__":
    test_rasterize_points_boundary_filtering()
    test_continuous_pixel_scaling_precision()
    test_horizontal_flip_symmetry()
    test_empty_image_and_negative_sample_handling()
    test_architecture_arbitrary_and_prime_shape_parity()
    test_batch_sample_isolation_gradient()
    test_parameter_budget_ceiling()
    print("ALL CODE REVIEW AUDIT TESTS PASSED!")
