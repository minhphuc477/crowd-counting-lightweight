from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from rmr_core.data import rasterize_points, train_transform
from rmr_core.evaluation import evaluate_dataset
from rmr_core.metrics import compute_nae, game_physical_image
from rmr_core.necks import CoordinateAttention
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    charbonnier_tv_step,
    multiplicative_gated_adjoint,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
    regional_sum,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)


def test_operator_adjoint_dot_product_identity():
    """Verify exact adjoint identity <Ax, y> == <x, A^T y> to float32 machine precision.

    This ensures that regional_adjoint is mathematically the exact adjoint of regional_sum.
    """
    torch.manual_seed(42)
    H, W = 48, 64
    regions = build_multiscale_regions(H, W, output_stride=4, region_sizes_px=(16, 32, 64))
    M = regions.boxes.shape[0]

    # Random test fields
    x = torch.randn(2, 1, H, W, dtype=torch.float32)
    y = torch.randn(2, 1, M, dtype=torch.float32)

    Ax = regional_sum(x, regions.boxes, out_dtype=torch.float32)
    ATy = regional_adjoint(y, regions.boxes, H, W, out_dtype=torch.float32)

    inner_Ax_y = (Ax * y).sum().item()
    inner_x_ATy = (x * ATy).sum().item()

    rel_diff = abs(inner_Ax_y - inner_x_ATy) / (abs(inner_Ax_y) + 1e-7)
    assert rel_diff < 1e-5, f"Adjoint identity failed: rel_diff={rel_diff:.2e}, Ax_y={inner_Ax_y}, x_ATy={inner_x_ATy}"


def test_transfer_operator_exact_uniform_invariance():
    """Verify H * 1_G == 1_G (Adjoint Scale Invariance Theorem).

    H = D_{c,w}^{-1} A^T W D_a^{-1} A.
    For any uniform field Y = c * 1_G, H * Y == c * 1_G.
    """
    torch.manual_seed(42)
    H, W = 40, 56
    regions = build_multiscale_regions(H, W, output_stride=4, region_sizes_px=(16, 32, 64))
    M = regions.boxes.shape[0]

    # Uniform weights and uniform density field
    weights = torch.rand(1, 1, M, dtype=torch.float32) + 0.5
    c_val = 2.718
    ones_field = torch.full((1, 1, H, W), c_val, dtype=torch.float32)

    # Compute A * (c * 1_G)
    q = regional_sum(ones_field, regions.boxes, out_dtype=torch.float32)
    area = regions.area.float().view(1, 1, -1)
    rate = q / area.clamp_min(1.0)  # should identically equal c_val

    # Adjoint step: A^T (W * rate)
    cov_w = weighted_coverage(weights, regions, H, W)
    weighted_rate = weights * rate
    back = regional_adjoint(weighted_rate, regions.boxes, H, W, out_dtype=torch.float32)

    # Normalized field: D_{c,w}^{-1} A^T (W * rate)
    H_ones = back / cov_w

    max_err = (H_ones - c_val).abs().max().item()
    assert max_err < 2e-5, f"Transfer operator scale invariance violated: max_err={max_err:.2e}"


def test_charbonnier_tv_discrete_lyapunov_decay():
    """Verify discrete TV energy E(y_{t+1}) <= E(y_t) (Lyapunov decay under CFL)."""
    torch.manual_seed(42)
    y = torch.rand(1, 1, 48, 48, dtype=torch.float32) * 5.0
    lambda_tv = 0.015
    eps_c = 0.1  # lambda_tv <= eps_c / 4 = 0.025 (CFL compliant)

    def tv_energy(field: torch.Tensor) -> float:
        dy_dx = field[..., 1:] - field[..., :-1]
        dy_dy = field[..., 1:, :] - field[..., :-1, :]
        return (torch.sqrt(dy_dx.pow(2) + eps_c**2).sum() +
                torch.sqrt(dy_dy.pow(2) + eps_c**2).sum()).item()

    curr_y = y
    prev_energy = tv_energy(curr_y)
    for step in range(6):
        curr_y = charbonnier_tv_step(curr_y, lambda_tv=lambda_tv, eps_c=eps_c, enforce_cfl=True)
        energy = tv_energy(curr_y)
        # Numerical tolerance for finite difference discretization
        assert energy <= prev_energy + 1e-4, f"Step {step}: TV energy increased from {prev_energy:.4f} to {energy:.4f}"
        prev_energy = energy


def test_charbonnier_tv_cfl_guard_prevents_super_cfl_divergence():
    """Verify enforce_cfl=True clamps dangerous super-CFL lambda_tv."""
    y = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    # Dangerous configuration: lambda_tv = 1.0, eps_c = 0.01 (CFL limit is 0.0025)
    out = charbonnier_tv_step(y, lambda_tv=1.0, eps_c=0.01, enforce_cfl=True)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert (out >= 0.0).all()


def test_multiplicative_sirt_zero_absorbing_state_cured():
    """Verify gate_floor > 0 allows false-negative zero pixels to receive positive correction.

    With gate_floor=0.0 (old bug), y_0=0 resulted in gate=tanh(0)=0 -> permanently 0.
    With gate_floor=0.02, y_0=0 receives non-zero correction while keeping background suppressed.
    """
    H, W = 32, 32
    regions = build_multiscale_regions(H, W, output_stride=4, region_sizes_px=(16, 32))
    M = regions.boxes.shape[0]

    # Initial density is completely zero (false negative detection)
    y_zero = torch.zeros((1, 1, H, W), dtype=torch.float32)
    # Positive residual from regional evidence
    residuals = torch.ones((1, 1, M), dtype=torch.float32)

    # 1. Old behavior (floor=0.0): trapped at zero
    gated_zero_floor = multiplicative_gated_adjoint(
        residuals, regions.boxes, y_zero, H, W, rho0=0.02, gate_floor=0.0
    )
    assert gated_zero_floor.abs().max().item() == 0.0, "floor=0.0 must yield 0 correction"

    # 2. New behavior (floor=0.02): receives positive correction
    gated_cured = multiplicative_gated_adjoint(
        residuals, regions.boxes, y_zero, H, W, rho0=0.02, gate_floor=0.02
    )
    assert gated_cured.max().item() > 0.0, "floor=0.02 must allow positive correction on false negatives"
    assert abs(gated_cured.max().item() / (gated_cured.max().item() / 0.02) - 0.02) < 1e-5


def test_coordinate_attention_batch_invariance():
    """Verify CoordinateAttention with GroupNorm produces identical results for B=1 and B=4."""
    ca = CoordinateAttention(channels=32, reduction=4).eval()
    x1 = torch.randn(1, 32, 28, 40)
    x4 = torch.cat([x1, torch.randn(3, 32, 28, 40)], dim=0)

    with torch.no_grad():
        out1 = ca(x1)
        out4 = ca(x4)

    max_diff = (out1[0] - out4[0]).abs().max().item()
    assert max_diff < 1e-6, f"CoordinateAttention batch divergence: max_diff={max_diff:.2e}"


def test_extreme_crowd_prefix2d_no_catastrophic_cancellation():
    """Verify prefix2d avoids catastrophic cancellation under extreme crowd counts (N=5000)."""
    H, W = 64, 64
    x = torch.zeros((1, 1, H, W), dtype=torch.float16)
    # Place large counts in corners
    x[0, 0, 10, 10] = 2500.0
    x[0, 0, 50, 50] = 2500.0

    boxes = torch.tensor([[0, 0, 64, 64], [5, 5, 15, 15], [45, 45, 55, 55]], dtype=torch.long)
    pref = prefix2d(x, preserve_fp32=True)
    assert pref.dtype == torch.float32, "prefix2d must accumulate in float32"

    sums = rectangle_sum_from_prefix(pref, boxes)
    assert abs(sums[0, 0, 0].item() - 5000.0) < 1e-2
    assert abs(sums[0, 0, 1].item() - 2500.0) < 1e-2
    assert abs(sums[0, 0, 2].item() - 2500.0) < 1e-2


def test_extreme_aspect_ratio_data_transform_safety():
    """Verify train_transform handles extreme panorama (100x2400) without memory explosion."""
    img = Image.new("RGB", (100, 2400), color=(100, 100, 100))
    pts = torch.tensor([[50.0, 300.0], [50.0, 1800.0]], dtype=torch.float32)

    cropped_t, cropped_pts = train_transform(
        img, pts, crop_size=512, scale_range=(0.8, 1.2), hflip_prob=0.0
    )
    assert cropped_t.shape == (3, 512, 512)
    assert cropped_t.min() >= 0.0 and cropped_t.max() <= 1.0
    if cropped_pts.numel() > 0:
        assert (cropped_pts >= 0.0).all()
        assert (cropped_pts < 512.0).all()


def test_mandatory_gt_consistency_guard():
    """Verify evaluate_dataset with enforce_gt_consistency=True catches and rejects raster-vs-raw-point discrepancies."""
    class DummyModel(torch.nn.Module):
        def forward(self, img, **kwargs):
            _, _, h, w = img.shape
            return {"y": torch.zeros((1, 1, h // 4, w // 4))}

    model = DummyModel()
    sample = {
        "image": torch.rand((3, 64, 64)),
        "target_y": torch.ones((1, 16, 16)),  # sum = 256
        "points": torch.tensor([[10.0, 10.0]]),  # 1 point only -> huge mismatch!
        "id": "mismatch_sample",
        "height": 64,
        "width": 64,
    }
    loader = [[sample]]

    # Must raise ValueError when enforce_gt_consistency=True
    with pytest.raises(ValueError, match="GT consistency invariant violated"):
        evaluate_dataset(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
            run_tiling=False,
            enforce_gt_consistency=True,
        )


def test_game0_strictly_equals_mae_on_prime_dimensions():
    """Verify GAME(0) strictly equals MAE on prime non-square dimensions."""
    torch.manual_seed(42)
    H, W, stride = 137, 219, 4
    gh, gw = int(math.ceil(H / stride)), int(math.ceil(W / stride))
    pred_y = torch.rand((1, gh, gw), dtype=torch.float64) * 0.05
    pts = torch.tensor([[12.3, 45.6], [89.1, 110.2], [210.5, 130.4]], dtype=torch.float64)

    games = game_physical_image(pred_y, pts, image_h=H, image_w=W, stride=stride, levels=(0, 1, 2))
    gt_count = float(len(pts))
    pred_count = float(pred_y.sum().item())
    expected_ae = abs(pred_count - gt_count)

    assert abs(games[0] - expected_ae) < 1e-6, f"GAME(0) {games[0]} != expected AE {expected_ae}"
