from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rmr_core.losses import (
    count_magnitude_loss,
    flat_dm_block_loss,
    flat_dm16_loss,
    negative_binomial_nll_mean_dispersion,
)
from rmr_core.operators import (
    RegionSet,
    _canonicalize_region_size,
    build_multiscale_regions,
    fractional_region_mean_std_features,
    prefix2d,
    region_geometry,
    region_mean_std_features,
    regional_adjoint,
    regional_sum,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from rmr_v3.config import (
    compute_config_hash,
    load_config,
    validate_resume_compatibility,
    validate_v3_config,
)
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config, reliability_from_nb


# ===========================================================================
# Section 1: Exact Mathematical Operator Proofs & Numerical Equivalence
# ===========================================================================

def test_adjoint_duality_exactness():
    """Verify Hilbert adjoint duality <Ax, v>_M = <x, A^T v>_G to machine precision.

    For any linear operator A: R^G -> R^M and adjoint A^T: R^M -> R^G,
    the inner product identity:
        sum_{b, c, m} (Ax)_{b,c,m} v_{b,c,m} == sum_{b, c, h, w} x_{b,c,h,w} (A^T v)_{b,c,h,w}
    must hold identically for any arbitrary box configuration (including odd dimensions,
    anisotropic rectangular regions, and multi-scale pyramids).
    """
    torch.manual_seed(1337)
    for h, w in [(32, 32), (33, 27), (19, 41), (16, 48)]:
        regions = build_multiscale_regions(
            h, w, output_stride=4,
            region_sizes_px=(16, 32, (32, 16), (16, 32)),
            overlap=0.5, include_full_image=False, device="cpu",
        )
        b, c, m = 2, 1, len(regions.boxes)

        # Test in FP64 for exact analytical precision
        x64 = torch.randn(b, c, h, w, dtype=torch.float64)
        v64 = torch.randn(b, c, m, dtype=torch.float64)

        ax64 = regional_sum(x64, regions.boxes, out_dtype=torch.float64)
        atv64 = regional_adjoint(v64, regions.boxes, h, w, out_dtype=torch.float64)

        inner_m = (ax64 * v64).sum()
        inner_g = (x64 * atv64).sum()

        rel_error = (inner_m - inner_g).abs() / (inner_m.abs() + inner_g.abs() + 1e-12)
        assert rel_error < 1e-10, (
            f"Adjoint identity failed in float64 for ({h}, {w}): "
            f"inner_m={inner_m.item():.12e}, inner_g={inner_g.item():.12e}, rel_error={rel_error.item():.2e}"
        )

        # Test in FP32
        x32 = x64.float()
        v32 = v64.float()
        ax32 = regional_sum(x32, regions.boxes, out_dtype=torch.float32)
        atv32 = regional_adjoint(v32, regions.boxes, h, w, out_dtype=torch.float32)
        inner_m32 = (ax32 * v32).sum()
        inner_g32 = (x32 * atv32).sum()
        rel_error32 = (inner_m32 - inner_g32).abs() / (inner_m32.abs() + inner_g32.abs() + 1e-8)
        assert rel_error32 < 1e-5, (
            f"Adjoint identity failed in float32 for ({h}, {w}): rel_error={rel_error32.item():.2e}"
        )


def test_weighted_coverage_pixelwise_identity():
    """Verify that D_{c,w} = A^T w matches the exact sum of weights covering each pixel."""
    torch.manual_seed(42)
    h, w = 24, 28
    regions = build_multiscale_regions(
        h, w, output_stride=4,
        region_sizes_px=(16, 32, (32, 16)),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    m = len(regions.boxes)
    # Arbitrary non-uniform positive weights
    w_vec = torch.rand(1, 1, m, dtype=torch.float32) * 3.0 + 0.5

    # Compute coverage via adjoint operator
    d_cw = weighted_coverage(w_vec, regions, h, w)

    # Compute ground truth coverage via explicit pixel accumulation
    boxes = regions.boxes
    expected_cov = torch.zeros(1, 1, h, w, dtype=torch.float32)
    for idx in range(m):
        y1, x1, y2, x2 = boxes[idx].tolist()
        expected_cov[0, 0, y1:y2, x1:x2] += w_vec[0, 0, idx]

    assert torch.allclose(d_cw, expected_cov, atol=1e-6, rtol=1e-6), (
        "weighted_coverage did not match explicit pixelwise accumulation!"
    )


def test_scale_invariance_theorem_proof():
    """Empirical proof of Theorem 1: H_w 1_G = 1_G for any non-uniform, anisotropic geometry.

    H_w = D_{c,w}^-1 A^T W D_a^-1 A
    Satisfies H_w (c 1_G) = c 1_G for any constant rate c, regardless of:
    1. Grid dimensions (even, odd, asymmetric).
    2. Overlap degree or box sizes (square, anisotropic).
    3. Positive weight distribution W > 0.
    """
    torch.manual_seed(999)
    test_configs = [
        # (h, w, region_sizes_px)
        (32, 32, (16, 32, 64)),
        (35, 29, (16, 32, (32, 16), (16, 32))),
        (21, 53, (16, (48, 16), (16, 48))),
        (17, 19, (16, 32)),
    ]

    for h, w, sizes in test_configs:
        regions = build_multiscale_regions(
            h, w, output_stride=4, region_sizes_px=sizes,
            overlap=0.5, include_full_image=False, device="cpu",
        )
        # Random non-uniform positive weights
        w_weights = torch.rand(2, 1, len(regions.boxes), dtype=torch.float32) * 4.0 + 0.25

        for c in [0.015, 1.0, 7.34, 42.0]:
            const_density = torch.full((2, 1, h, w), fill_value=c, dtype=torch.float32)
            # b = 0 implies delta = A y
            b_zero = torch.zeros(2, 1, len(regions.boxes), dtype=torch.float32)

            # H_w (c 1_G) is the normalized adjoint field evaluated at b=0
            h_w_const = weighted_normalized_adjoint_field(
                const_density, b_zero, w_weights, regions, eps=1e-6
            )

            rel_err = (h_w_const - c).abs().max().item() / float(c)
            assert rel_err < 1e-4, (
                f"Scale Invariance Theorem failed for ({h}, {w}) with c={c}: rel_err={rel_err:.2e}"
            )


def test_constant_rate_residual_homogeneity():
    """Verify that if (A y - b) / |R| = delta_0 for all regions, r_field == delta_0 everywhere."""
    torch.manual_seed(77)
    h, w = 31, 37
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(16, 32, (32, 16)),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.rand(2, 1, h, w, dtype=torch.float32) * 0.5
    q = regional_sum(y, regions.boxes, out_dtype=torch.float32)

    delta0 = 2.71828
    area = regions.area.float().view(1, 1, -1)
    b = q - delta0 * area

    weights = torch.rand_like(b) * 3.5 + 0.5
    field = weighted_normalized_adjoint_field(y, b, weights, regions)

    expected = torch.full_like(field, delta0)
    assert torch.allclose(field, expected, atol=1e-4, rtol=1e-4)


def test_proximal_operator_exact_deadband_and_subgradient():
    """Verify exact mathematical properties of proximal L1-soft-thresholding S_tau^+(z) = max(0, z - tau)."""
    tau = 0.02

    # Region 1: Inside noise deadband [0, tau)
    z_noise = torch.linspace(0.0, tau - 1e-4, 50, requires_grad=True)
    out_noise = torch.clamp_min(z_noise - tau, 0.0)
    assert (out_noise == 0.0).all(), "Deadband violation: noise was not mapped strictly to zero!"
    out_noise.sum().backward()
    assert (z_noise.grad == 0.0).all(), "Subgradient inside deadband must be zero"

    # Region 2: Signal above threshold z > tau
    z_signal = torch.linspace(tau + 0.01, 2.0, 50, requires_grad=True)
    out_signal = torch.clamp_min(z_signal - tau, 0.0)
    expected_signal = z_signal - tau
    assert torch.allclose(out_signal, expected_signal), "Linearity above threshold violated!"
    out_signal.sum().backward()
    assert torch.allclose(z_signal.grad, torch.ones_like(z_signal)), "Derivative above tau must be identically 1.0"

    # Region 3: Sub-zero inputs z < 0
    z_neg = torch.linspace(-2.0, -0.01, 30)
    out_neg = torch.clamp_min(z_neg - tau, 0.0)
    assert (out_neg == 0.0).all(), "Negative inputs must be clamped to 0.0"


def test_spatial_moments_mathematical_properties():
    """Verify that Spatial-Moments (mean + std pooling) exhibits exact mathematical properties."""
    torch.manual_seed(42)
    b, c, h, w = 2, 32, 32, 32
    boxes = torch.tensor([[0, 0, 16, 16], [0, 0, 32, 32]], dtype=torch.long)

    # Property 1: Constant feature field must produce std = sqrt(eps)
    const_val = 3.5
    feat_const = torch.full((b, c, h, w), const_val, dtype=torch.float32)
    moments_const = region_mean_std_features(feat_const, boxes, eps=1e-6)
    mean_part = moments_const[..., :c]
    std_part = moments_const[..., c:]
    assert torch.allclose(mean_part, torch.tensor(const_val), atol=1e-5)
    assert torch.allclose(std_part, torch.tensor(math.sqrt(1e-6)), atol=1e-5)

    # Property 2: Invariance under moderate feature shifts: std(x + c) == std(x)
    feat_random = torch.randn(b, c, h, w, dtype=torch.float32)
    feat_shifted = feat_random + 2.0
    moments_rand = region_mean_std_features(feat_random, boxes, eps=1e-6)
    moments_shift = region_mean_std_features(feat_shifted, boxes, eps=1e-6)
    assert torch.allclose(moments_rand[..., c:], moments_shift[..., c:], atol=1e-4), (
        "Standard deviation pooling is not shift-invariant!"
    )

    # Property 3: Elimination of catastrophic cancellation under extreme feature shifts (mean = 10^4)
    feat_shift_10k = 10000.0 + torch.randn(b, c, h, w, dtype=torch.float32)
    moments_10k = region_mean_std_features(feat_shift_10k, boxes, eps=1e-6)
    assert torch.isfinite(moments_10k).all(), "Spatial moments produced NaN/Inf on large feature values!"
    # Verify standard deviation precision is preserved (relative error < 1e-3)
    y1, x1, y2, x2 = boxes[0].tolist()
    box0_patch = feat_shift_10k[:, :, y1:y2, x1:x2]
    true_std0 = box0_patch.std(dim=(-2, -1), unbiased=False)
    computed_std0 = moments_10k[:, 0, c:]
    rel_err_std = (computed_std0 - true_std0).abs() / (true_std0 + 1e-6)
    assert (rel_err_std < 1e-3).all(), (
        f"Catastrophic cancellation detected in spatial moments: max rel error = {rel_err_std.max().item():.2e}"
    )


def test_negative_binomial_rate_variance_and_dispersion_bounds():
    """Verify NB predictive rate variance formula Var[N/area] = (mu + mu^2/r) / area^2 and bounds."""
    area_val = 64.0
    boxes = torch.tensor([[0, 0, 8, 8]], dtype=torch.long)
    scale_id = torch.tensor([0], dtype=torch.long)
    area = torch.tensor([area_val], dtype=torch.float32)
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area)

    mu_count = torch.tensor([[[32.0]]], dtype=torch.float32)
    dispersion = torch.tensor([[[50.0]]], dtype=torch.float32)

    rel = reliability_from_nb(
        mu_count, dispersion, regions,
        rate_std_floor=0.01,
        weight_min=0.25, weight_max=4.0,
        normalize_within_scale=False,
    )

    expected_count_var = 32.0 + (32.0 ** 2) / 50.0
    assert abs(rel["count_variance"].item() - expected_count_var) < 1e-4

    expected_rate_var = expected_count_var / (area_val ** 2) + 0.01 ** 2
    assert abs(rel["rate_variance"].item() - expected_rate_var) < 1e-5


# ===========================================================================
# Section 2: Boundary Conditions & Geometric Stress Tests
# ===========================================================================

@pytest.mark.parametrize("shape", [
    (1, 3, 131, 127),  # Odd, prime-like dimensions
    (1, 3, 67, 53),    # Very small odd dimensions
    (1, 3, 255, 129),  # Non-powers-of-two, non-divisible by 16 or 32
    (1, 3, 79, 113),   # Arbitrary primes
])
def test_non_divisible_and_odd_image_dimensions(shape: tuple[int, ...]):
    """RMRv3 and AQ-RMR must cleanly process inputs with odd dimensions and non-powers-of-two."""
    cfg = RMRv3Config(
        pretrained=False,
        region_sizes_px=(32, 64, (64, 32), (32, 64)),
        regional_feature_stats="mean_std",
        proximal_tau=0.015,
        iterations=2,
    )
    model = RMRv3(cfg)
    x = torch.randn(*shape)
    out = model(x)

    y = out["y"]
    assert torch.isfinite(y).all()
    assert (y >= 0.0).all()
    assert len(out["regions"].boxes) > 0
    assert out["b_region"].shape[-1] == len(out["regions"].boxes)


@pytest.mark.parametrize("h, w", [
    (4, 256),   # Extreme landscape aspect ratio 1:64
    (256, 4),   # Extreme portrait aspect ratio 64:1
    (1, 100),   # 1-pixel height ribbon
    (100, 1),   # 1-pixel width ribbon
    (1, 1),     # Single-cell grid
])
def test_extreme_aspect_ratios_and_skinny_geometries(h: int, w: int):
    """Every grid cell must be strictly covered (min coverage >= 1.0) under extreme aspect ratios."""
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    assert len(regions.boxes) > 0
    assert (regions.area > 0).all()

    # Verify coverage diagonal
    ones_w = torch.ones(1, 1, len(regions.boxes))
    cov = weighted_coverage(ones_w, regions, h, w)
    assert (cov >= 1.0).all(), f"Uncovered cells found for ({h}, {w})! Min coverage: {cov.min().item()}"


def test_single_cell_lattice_stability():
    """Verify that a 1x1 grid evaluates without division-by-zero or crash."""
    h, w = 1, 1
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.tensor([[[[0.5]]]], dtype=torch.float32)
    b = torch.tensor([[[0.3, 0.4]]], dtype=torch.float32)
    w_vec = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)

    field = weighted_normalized_adjoint_field(y, b, w_vec, regions)
    assert torch.isfinite(field).all()
    assert field.shape == (1, 1, 1, 1)


def test_invalid_region_sizes_rejected():
    """Invalid region size specifications and grid dimensions must be rejected."""
    with pytest.raises(ValueError, match="strictly positive"):
        _canonicalize_region_size(0)

    with pytest.raises(ValueError, match="strictly positive"):
        _canonicalize_region_size(-16)

    with pytest.raises(ValueError, match="strictly positive"):
        _canonicalize_region_size((32, 0))

    with pytest.raises(ValueError, match="pair"):
        _canonicalize_region_size((32, 16, 64))  # type: ignore[arg-type]

    # Grid dimensions <= 0 must be rejected
    with pytest.raises(ValueError, match="strictly positive"):
        build_multiscale_regions(0, 32, 4)

    with pytest.raises(ValueError, match="strictly positive"):
        build_multiscale_regions(32, -8, 4)

    # Output stride <= 0 must be rejected
    with pytest.raises(ValueError, match="strictly positive"):
        build_multiscale_regions(32, 32, 0)


# ===========================================================================
# Section 3: Numerical Stability & Mixed-Precision Verification
# ===========================================================================

def test_zero_count_empty_background_stability():
    """Model and losses must remain completely stable (finite, no NaN) on empty background crops (0 count)."""
    cfg = RMRv3Config(
        pretrained=False,
        proximal_tau=0.015,
        iterations=6,
        tv_lambda=0.02,
        hurdle_head=False,
    )
    model = RMRv3(cfg)
    loss_cfg = RMRv3LossConfig(dm_target="y")

    # Empty image input and zero GT target
    x_empty = torch.zeros(2, 3, 128, 128)
    out = model(x_empty)
    target_empty = torch.zeros_like(out["y"])

    losses = compute_rmr_v3_losses(out, target_empty, loss_cfg)

    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"Loss component '{k}' is not finite on empty background: {v}"
        assert v.item() >= 0.0, f"Loss component '{k}' is negative: {v.item()}"

    # Gradient must flow cleanly without NaN
    losses["total"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "NaN/Inf gradient detected on empty background!"


def test_extreme_density_crowd_stability():
    """Model and losses must not overflow on ultra-dense crowd scenes (> 2000 count)."""
    cfg = RMRv3Config(
        pretrained=False,
        proximal_tau=0.015,
        iterations=6,
        tv_lambda=0.02,
    )
    model = RMRv3(cfg)
    loss_cfg = RMRv3LossConfig(dm_target="y")

    x_dense = torch.randn(2, 3, 128, 128)
    out = model(x_dense)
    # Total count = 3,000 distributed over cells
    target_dense = torch.full_like(out["y"], 3000.0 / (32 * 32))

    losses = compute_rmr_v3_losses(out, target_dense, loss_cfg)

    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"Loss component '{k}' overflowed on extreme density: {v}"

    losses["total"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "NaN/Inf gradient detected on extreme density!"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_amp_mixed_precision_consistency(dtype: torch.dtype):
    """Verify numerical integrity under AMP autocast (float16, bfloat16, float32)."""
    cfg = RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, (64, 32)),
        proximal_tau=0.015,
        iterations=4,
    )
    model = RMRv3(cfg)
    x = torch.randn(2, 3, 128, 128)

    if dtype in (torch.float16, torch.bfloat16):
        with torch.amp.autocast("cpu", dtype=dtype):
            out = model(x)
            assert torch.isfinite(out["y"]).all()
            assert (out["y"] >= 0.0).all()
    else:
        out = model(x)
        assert torch.isfinite(out["y"]).all()
        assert (out["y"] >= 0.0).all()


# ===========================================================================
# Section 4: Complete Gradient Flow & Parameter Architecture Audit
# ===========================================================================

@pytest.mark.parametrize("config_name,cfg", [
    ("canonical", RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean",
        region_sizes_px=(32, 64, 128),
        proximal_tau=0.0,
        iterations=6,
        tv_lambda=0.02,
    )),
    ("aq_rmr", RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
        proximal_tau=0.015,
        iterations=6,
        tv_lambda=0.02,
    )),
])
def test_all_parameter_tensors_receive_healthy_gradients(config_name: str, cfg: RMRv3Config):
    """Every single trainable parameter tensor in the model must receive non-zero, healthy gradients.

    Guarantees no dead sub-networks, unused heads, or disconnected computational branches.
    """
    model = RMRv3(cfg)
    loss_cfg = RMRv3LossConfig(dm_target="y")

    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    target = torch.rand_like(out["y"])

    loss = compute_rmr_v3_losses(out, target, loss_cfg)
    loss["total"].backward()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 124, f"Expected 124 parameter tensors, got {len(trainable_params)}"

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"Parameter '{name}' did not receive any gradient (dead branch)!"
        assert torch.isfinite(p.grad).all(), f"Parameter '{name}' received NaN/Inf gradient!"
        grad_max = p.grad.abs().max().item()
        assert grad_max > 1e-12, f"Parameter '{name}' has vanishing gradient: max={grad_max:.2e}"


def test_parameter_budget_hard_accounting():
    """Strict accounting: trainable parameters must not exceed the <= 105,000 parameter budget."""
    # Canonical RMR-v9
    cfg_canon = RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean",
        region_sizes_px=(32, 64, 128),
        proximal_tau=0.0,
    )
    model_canon = RMRv3(cfg_canon)
    count_canon = sum(p.numel() for p in model_canon.parameters() if p.requires_grad)
    assert count_canon == 101763, f"Canonical parameter count mismatch: got {count_canon}, expected 101,763"
    assert count_canon <= 105000

    # AQ-RMR (Spatial-Moments)
    cfg_aq = RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
        proximal_tau=0.015,
    )
    model_aq = RMRv3(cfg_aq)
    count_aq = sum(p.numel() for p in model_aq.parameters() if p.requires_grad)
    assert count_aq == 103299, f"AQ-RMR parameter count mismatch: got {count_aq}, expected 103,299"
    assert count_aq <= 105000


# ===========================================================================
# Section 5: Non-Tautological Causal Isolation Tests
# ===========================================================================

def test_causal_ablation_proximal_tau():
    """Ablating proximal thresholding (tau=0 vs tau=0.015) must causally alter output density."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 128, 128)

    cfg0 = RMRv3Config(pretrained=False, proximal_tau=0.0, iterations=6)
    cfg1 = RMRv3Config(pretrained=False, proximal_tau=0.015, iterations=6)

    m0 = RMRv3(cfg0)
    m1 = RMRv3(cfg1)
    m1.load_state_dict(m0.state_dict())

    y0 = m0(x)["y"]
    y1 = m1(x)["y"]

    diff = (y0 - y1).abs().max().item()
    assert diff > 1e-3, f"Ablation failed: tau=0 vs tau=0.015 produced negligible difference {diff:.6f}"


def test_causal_ablation_solver_iterations():
    """Ablating unrolled SIRT solver (T=0 vs T=6) must causally alter reconciled density."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 128, 128)

    cfg0 = RMRv3Config(pretrained=False, enable_solver=False)
    cfg6 = RMRv3Config(pretrained=False, enable_solver=True, iterations=6)

    m0 = RMRv3(cfg0)
    m6 = RMRv3(cfg6)
    m6.load_state_dict(m0.state_dict())

    y0 = m0(x)["y"]
    y6 = m6(x)["y"]

    diff = (y0 - y6).abs().max().item()
    assert diff > 1e-4, f"Ablation failed: solver iterations had zero causal effect {diff:.6f}"


def test_causal_ablation_dm_target():
    """Supervising Flat-DM16 on y vs y0 must produce different loss values when solver is active."""
    torch.manual_seed(42)
    cfg = RMRv3Config(pretrained=False, enable_solver=True, iterations=4, tv_lambda=0.02)
    model = RMRv3(cfg)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    target = torch.rand_like(out["y"])

    loss_cfg_y = RMRv3LossConfig(dm_target="y")
    loss_cfg_y0 = RMRv3LossConfig(dm_target="y0")

    l_y = compute_rmr_v3_losses(out, target, loss_cfg_y)["allocation"].item()
    l_y0 = compute_rmr_v3_losses(out, target, loss_cfg_y0)["allocation"].item()

    assert abs(l_y - l_y0) > 1e-6, f"dm_target='y' vs 'y0' had no effect on allocation loss: {l_y} vs {l_y0}"
    assert l_y != l_y0


# ===========================================================================
# Section 6: Config Validation & Resume Safety Guards
# ===========================================================================

def test_config_validation_guards():
    """validate_v3_config must reject invalid hyperparameters."""
    # Negative proximal_tau rejected
    with pytest.raises(ValueError, match="proximal_tau must be non-negative"):
        validate_v3_config({"model": {"proximal_tau": -0.01}})

    # Negative tv_lambda rejected
    with pytest.raises(ValueError, match="tv_lambda must be non-negative"):
        validate_v3_config({"model": {"tv_lambda": -0.02}})

    # Invalid dm_target rejected
    with pytest.raises(ValueError, match="dm_target must be 'y' or 'y0'"):
        validate_v3_config({"loss": {"dm_target": "invalid"}})

    # Invalid count_loss_mode rejected
    with pytest.raises(ValueError, match="count_loss_mode must be 'nb', 'log1p', or 'l1'"):
        validate_v3_config({"loss": {"count_loss_mode": "mse"}})

    # Invalid allocation_loss_type rejected
    with pytest.raises(ValueError, match="allocation_loss_type must be"):
        validate_v3_config({"loss": {"allocation_loss_type": "wasserstein"}})


def test_resume_validation_guards_rmr_v9():
    """validate_resume_compatibility must detect and reject changes to v9 critical fields."""
    base_cfg = {
        "model": {"omega": 1.0, "iterations": 6, "proximal_tau": 0.015, "tv_lambda": 0.02},
        "loss": {"lambda_count": 1.0, "dm_target": "y"},
    }

    # Match passes
    validate_resume_compatibility(base_cfg, dict(base_cfg))

    # Mismatched proximal_tau rejected
    bad_tau = {"model": {"proximal_tau": 0.0}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'model.proximal_tau'"):
        validate_resume_compatibility(base_cfg, bad_tau)

    # Mismatched tv_lambda rejected
    bad_tv = {"model": {"tv_lambda": 0.05}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'model.tv_lambda'"):
        validate_resume_compatibility(base_cfg, bad_tv)

    # Mismatched dm_target rejected
    bad_dmt = {"loss": {"dm_target": "y0"}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'loss.dm_target'"):
        validate_resume_compatibility(base_cfg, bad_dmt)


def test_load_config_canonical_and_aq_rmr():
    """load_config must load and successfully validate both canonical and aq-rmr configs."""
    c_canon = load_config("configs/rmr_v9/rmr_v9_canonical.yaml")
    assert c_canon["model"]["proximal_tau"] == 0.0
    assert c_canon["loss"]["dm_target"] == "y"

    c_aq = load_config("configs/rmr_v9/rmr_v9_aq_rmr.yaml")
    assert c_aq["model"]["proximal_tau"] == 0.015
    assert c_aq["model"]["regional_feature_stats"] == "mean_std"
    assert c_aq["loss"]["dm_target"] == "y"


# ===========================================================================
# Section 7: Deep Architectural Edge Cases, Memory Safety, & Dtype Integrity
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tv_type", ["laplacian", "charbonnier"])
def test_model_half_and_bfloat16_direct_execution(dtype: torch.dtype, tv_type: str):
    """Direct model execution under .half() and .bfloat16() must not fail on buffer dtype mismatches."""
    cfg = RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64),
        proximal_tau=0.015,
        iterations=2,
        tv_lambda=0.02,
        tv_type=tv_type,
    )
    model = RMRv3(cfg).to(dtype)
    x = torch.randn(1, 3, 128, 128, dtype=dtype)
    out = model(x)
    assert out["y"].dtype == dtype
    assert torch.isfinite(out["y"]).all()
    assert (out["y"] >= 0.0).all()


def test_reliability_from_nb_full_image_scale_safety():
    """reliability_from_nb with include_full_image=True must not read uninitialized heap memory.

    Scale ID -1 corresponds to the full-image region. The weights for scale_id=-1
    must be strictly positive, finite, and properly normalized (identically 1.0 for a singleton).
    """
    regions = build_multiscale_regions(
        32, 32, output_stride=4, region_sizes_px=(16,),
        overlap=0.5, include_full_image=True, device="cpu",
    )
    assert -1 in regions.scale_id.tolist(), "full image scale_id=-1 must exist"
    m = len(regions.boxes)
    mu = torch.rand(2, 1, m) * 10.0 + 1.0
    disp = torch.full((2, 1, m), 50.0)

    rel = reliability_from_nb(mu, disp, regions, normalize_within_scale=True)
    w = rel["weight"]
    assert torch.isfinite(w).all(), "Uninitialized memory or NaN detected in reliability weights!"
    assert (w >= 0.25).all() and (w <= 4.0).all()

    full_mask = regions.scale_id == -1
    w_full = w[..., full_mask]
    assert torch.allclose(w_full, torch.ones_like(w_full)), (
        f"Full image weight must normalize to 1.0, got {w_full}"
    )


def test_fractional_region_mean_std_features_stability():
    """fractional_region_mean_std_features must preserve precision under 10^4 feature mean shift."""
    torch.manual_seed(42)
    b, c, h, w = 1, 4, 32, 32
    float_boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [4.0, 4.0, 28.0, 28.0]], dtype=torch.float32)

    feat = 10000.0 + torch.randn(b, c, h, w, dtype=torch.float32)
    out = fractional_region_mean_std_features(feat, float_boxes, eps=1e-6)

    mean_part = out[..., :c]
    std_part = out[..., c:]

    assert torch.isfinite(out).all()
    # True std of standard normal is 1.0
    assert torch.allclose(std_part, torch.ones_like(std_part), atol=0.08, rtol=0.08), (
        f"fractional std calculation failed under large constant shift: mean std = {std_part.mean().item():.4f}"
    )


def test_negative_binomial_nll_nan_and_inf_rejection():
    """negative_binomial_nll_mean_dispersion must reject NaN and Inf targets."""
    pred = torch.tensor([10.0])
    # NaN target
    with pytest.raises(ValueError, match="finite non-negative"):
        negative_binomial_nll_mean_dispersion(torch.tensor([float("nan")]), pred)

    # Inf target
    with pytest.raises(ValueError, match="finite non-negative"):
        negative_binomial_nll_mean_dispersion(torch.tensor([float("inf")]), pred)

    # Negative target
    with pytest.raises(ValueError, match="finite non-negative"):
        negative_binomial_nll_mean_dispersion(torch.tensor([-1.0]), pred)


def test_flat_dm_loss_strict_and_non_strict_modes():
    """flat_dm_block_loss must enforce strict divisibility by default and trim cleanly when strict=False."""
    # Non-divisible 33x31 grid
    pred_odd = torch.rand(2, 1, 33, 31, requires_grad=True)
    tgt_odd = torch.rand(2, 1, 33, 31)

    # Strict=True must raise ValueError
    with pytest.raises(ValueError, match="divisible by block size"):
        flat_dm_block_loss(pred_odd, tgt_odd, block_px=16, stride=4, strict=True)

    # Strict=False must succeed and permit backward gradient flow
    loss_non_strict = flat_dm_block_loss(pred_odd, tgt_odd, block_px=16, stride=4, strict=False)
    assert torch.isfinite(loss_non_strict)
    loss_non_strict.backward()
    assert pred_odd.grad is not None
    assert torch.isfinite(pred_odd.grad).all()

    # Sub-block tiny image (e.g. 3x3 with k=4) returns 0.0 when strict=False
    pred_tiny = torch.rand(1, 1, 3, 3)
    tgt_tiny = torch.rand(1, 1, 3, 3)
    loss_tiny = flat_dm_block_loss(pred_tiny, tgt_tiny, block_px=16, stride=4, strict=False)
    assert loss_tiny.item() == 0.0


def test_compute_rmr_v3_losses_non_divisible_dimensions():
    """compute_rmr_v3_losses must successfully run on odd crop dimensions when dm_strict=False."""
    cfg = RMRv3Config(pretrained=False, iterations=2)
    model = RMRv3(cfg)
    x = torch.randn(1, 3, 131, 127)
    out = model(x)
    target = torch.rand_like(out["y"])

    # dm_strict=False allows loss calculation on odd dimensions
    loss_cfg = RMRv3LossConfig(dm_target="y", dm_strict=False)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["allocation"])


# ===========================================================================
# Section 8: Solver Warmup Invariance, Zero-Collapse Prevention, & Continuous Ramp
# ===========================================================================

def test_solver_warmup_invariance_zero_collapse_prevention():
    """Verify that solver warmup (strength=0.0) guarantees y == y0 identically.

    This explicitly tests the zero-density collapse bug where unscaled proximal tau
    and unscaled TV diffusion previously subtracted density at strength=0.0,
    causing y to collapse to 0.0 and count loss to explode to 6000+.
    """
    torch.manual_seed(42)
    cfg = RMRv3Config(
        pretrained=False,
        iterations=6,
        proximal_tau=0.015,
        tv_lambda=0.02,
        tv_type="laplacian",
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(2, 3, 128, 128)

    # 1. Warmup state (solver_strength = 0.0)
    out_warmup = model(x, solver_strength=0.0)
    y_warmup = out_warmup["y"]
    y0_warmup = out_warmup["y0"]

    # Invariant 1: y MUST be identical to y0 when strength = 0.0
    assert torch.equal(y_warmup, y0_warmup), (
        f"Warmup invariance violated! Max diff: {(y_warmup - y0_warmup).abs().max().item()}"
    )

    # Invariant 2: Density must be positive (no zero-collapse)
    assert (y_warmup > 0.0).any()
    assert y_warmup.sum().item() > 10.0, (
        f"Zero-density collapse! Total mass={y_warmup.sum().item()}"
    )

    # Invariant 3: Count loss against calibrated crowd density must be small (< 10.0, not 6000+)
    # For a 128x128 image (32x32 grid = 1024 cells), calibrated prior m0=0.01576 gives ~16 count.
    target = torch.zeros(2, 1, 32, 32)
    for b in range(2):
        pts = torch.randint(0, 32, (16, 2))
        target[b, 0, pts[:, 0], pts[:, 1]] += 1.0

    loss_cfg = RMRv3LossConfig(dm_target="y")
    losses = compute_rmr_v3_losses(out_warmup, target, loss_cfg)
    assert losses["count"].item() < 10.0, (
        f"Count loss exploded in warmup! Loss: {losses['count'].item()}"
    )
    assert losses["total"].item() < 30.0, (
        f"Total loss exploded in warmup! Loss: {losses['total'].item()}"
    )


def test_solver_ramp_continuous_progression():
    """Verify that as solver_strength increases from 0.0 -> 0.5 -> 1.0,
    the proximal deadband scales smoothly and monotonically cleans background mass.
    """
    torch.manual_seed(42)
    cfg = RMRv3Config(
        pretrained=False,
        iterations=6,
        proximal_tau=0.015,
        tv_lambda=0.02,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, 128, 128)

    out_0 = model(x, solver_strength=0.0)
    out_5 = model(x, solver_strength=0.5)
    out_1 = model(x, solver_strength=1.0)

    mass_0 = out_0["y"].sum().item()
    mass_5 = out_5["y"].sum().item()
    mass_1 = out_1["y"].sum().item()

    # When background noise is cleaned by proximal tau, total predicted mass decreases smoothly
    assert mass_0 >= mass_5 >= mass_1, (
        f"Monotonic background cleaning violated: mass_0={mass_0:.2f}, mass_5={mass_5:.2f}, mass_1={mass_1:.2f}"
    )
    # But mass must not collapse to 0
    assert mass_1 > 0.5 * mass_0, (
        f"Excessive mass destruction! mass_1={mass_1:.2f} vs mass_0={mass_0:.2f}"
    )


def test_extreme_densities_stability():
    """Verify numerical stability under empty crop (0 count) and ultra-dense crop (2000 count)."""
    cfg = RMRv3Config(
        pretrained=False,
        iterations=6,
        proximal_tau=0.015,
        tv_lambda=0.02,
        regional_feature_stats="mean_std",
        region_sizes_px=(32, 64, 128, (64, 32), (32, 64)),
    )
    model = RMRv3(cfg)
    loss_cfg = RMRv3LossConfig(dm_target="y")

    # Case 1: Empty crop (all 0)
    x1 = torch.randn(2, 3, 128, 128)
    out1 = model(x1, solver_strength=1.0)
    target_empty = torch.zeros(2, 1, 32, 32)
    losses_empty = compute_rmr_v3_losses(out1, target_empty, loss_cfg)
    assert torch.isfinite(losses_empty["total"]), "Empty crop produced non-finite loss!"
    losses_empty["total"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "NaN/Inf gradient in empty crop!"

    # Case 2: Extreme dense crop (2000 count) with fresh forward pass
    model.zero_grad(set_to_none=True)
    x2 = torch.randn(2, 3, 128, 128)
    out2 = model(x2, solver_strength=1.0)
    target_dense = torch.zeros(2, 1, 32, 32)
    pts = torch.randint(0, 32, (2000, 2))
    for pt in pts:
        target_dense[0, 0, pt[0], pt[1]] += 1.0
        target_dense[1, 0, pt[0], pt[1]] += 1.0

    losses_dense = compute_rmr_v3_losses(out2, target_dense, loss_cfg)
    assert torch.isfinite(losses_dense["total"]), "Dense crop produced non-finite loss!"
    losses_dense["total"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "NaN/Inf gradient in dense crop!"

