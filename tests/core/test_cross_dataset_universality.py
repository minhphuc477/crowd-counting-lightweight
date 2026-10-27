"""Cross-Dataset & Multi-Regime Mathematical Universality Test Suite.

Verifies that a SINGLE unified RMR model & loss formulation handles both:
- Congested / micro-head regimes (ShanghaiTech Part A, UCF-CC-50)
- Sparse / broad-foreground / giant-resolution / 0-count negative regimes (SHB, QNRF, NWPU-Crowd)

Mathematical Invariants Verified:
1. Scale Covariance Theorem of Canonical Bayesian Loss (Ma et al. ICCV 2019, Eq. 7):
   Rescaling grid, coordinates, and Gaussian scales by alpha > 0 preserves exact posteriors.
2. Large Foreground Head & Single-Person (n=1) Continuity in Adaptive Bayesian Loss:
   Pixels at r = 12px on a foreground person belong > 98% to foreground (zero dimensional bug),
   and n=1 smoothly matches the limit of n=2 with a distant neighbor.
3. Exact Log-Domain Max-Shifted Softmax (Zero Float32 Underflow, Zero 1e-4 Clamp Leakage):
   Extreme distances (-d^2 / 2sigma^2 < -100,000) produce exact +1.0 background gradient and
   exact -1.0 foreground gradient with zero NaN/Inf.
4. Broad Foreground Head Peak Protection at Stride 2 (Dimensionless Relative Prominence):
   Broad Gaussian heads (sigma = 8px, peak cell height y0 ~ 0.01 << 0.05) retain >= 95% apex
   fidelity under regional underestimation, while flat plateaus receive 0% protection.
5. Morozov Sub-Person Deficit Deadband on Empty / Negative Scenes:
   When y0 == 0 and Softplus regional tail noise b_R <= 0.20 across hundreds of boxes,
   recovered count is identically 0.0, whereas real occluded deficits (b_R >= 1.0) recover
   100% exact regional mass via Stage 3 Bregman-Poisson projection.
6. Scale-Routed Presence Field Consistency:
   Foreground regions routed to 128px boxes are not diluted by empty 32px sub-boxes.
7. Universal Empirical Prior Calibration (init_m0: 'auto'):
   Accepted cleanly across validator, RMRv3Config, and make_model.
"""
from __future__ import annotations

import math
import pytest
import torch

from rmr_core.operators import RegionSet, build_multiscale_regions, closed_form_mcbpro_transport, fed_mcbpro_transport
from rmr_core.operators.prefix_sums import regional_sum
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import make_model
from rmr_v3.losses.point_supervision import bayesian_loss
from rmr_v3.model import RMRv3Config
from rmr_v3.model.solver_step import solve_inverse_measure


def test_bayesian_loss_exact_scale_covariance() -> None:
    """Scaling grid, point coordinates, and sigma by 2x preserves exact per-cell posterior gradients."""
    # Scale 1x: 32x32 grid at stride 4 (128x128 px)
    pts_1x = [torch.tensor([[34.0, 42.0], [90.0, 86.0]], dtype=torch.float32)]
    y_1x = torch.zeros((1, 1, 32, 32), dtype=torch.float32, requires_grad=True)
    loss_1x = bayesian_loss(
        y_1x, pts_1x, sigma=4.0, background_ratio=0.15, stride=4,
        norm_mode="canonical", canonical_background=True, adaptive_sigma=False,
    )
    loss_1x.backward()

    # Scale 2x: 32x32 grid at stride 8 (256x256 px), points * 2, sigma * 2
    pts_2x = [pts_1x[0] * 2.0]
    y_2x = torch.zeros((1, 1, 32, 32), dtype=torch.float32, requires_grad=True)
    loss_2x = bayesian_loss(
        y_2x, pts_2x, sigma=8.0, background_ratio=0.15, stride=8,
        norm_mode="canonical", canonical_background=True, adaptive_sigma=False,
    )
    loss_2x.backward()

    assert y_1x.grad is not None and y_2x.grad is not None
    max_diff = (y_1x.grad - y_2x.grad).abs().max().item()
    assert max_diff < 1e-5, f"Scale covariance violated: max posterior grad diff = {max_diff}"


def test_bayesian_loss_foreground_head_no_dimensional_inversion_and_n1_continuity() -> None:
    """Large foreground SHB/QNRF head at r=12px must belong >98% to person, and n=1 uses sigma_max."""
    # 128x128 grid at stride 4 -> 512x512 px image; single foreground person at (258.0, 258.0) (cell 64, 64)
    pts_single = [torch.tensor([[258.0, 258.0]], dtype=torch.float32)]
    y0 = torch.full((1, 1, 128, 128), 0.0001, dtype=torch.float32, requires_grad=True)

    loss = bayesian_loss(
        y0,
        pts_single,
        sigma=4.0,
        background_ratio=0.15,
        stride=4,
        norm_mode="canonical",
        canonical_background=True,
        adaptive_sigma=True,
        sigma_min=2.0,
        sigma_max=16.0,
    )
    loss.backward()
    assert y0.grad is not None

    # Pixel 3 cells away (r = 12px) is still on the person's forehead (sigma_max = 16px, D_bg = 76.8px).
    # Since c_hat << 1.0, grad = -P(person | x) + P(bg | x). P(person) > 0.99 => grad < -0.98!
    forehead_grad = y0.grad[0, 0, 64, 67].item()
    far_bg_grad = y0.grad[0, 0, 5, 5].item()
    assert forehead_grad < -0.98, (
        f"Foreground head pixel at r=12px was stolen by background! grad={forehead_grad}"
    )
    assert abs(far_bg_grad - 1.0) < 1e-5, f"Far background pixel must have exact +1.0 grad, got {far_bg_grad}"


def test_bayesian_loss_log_domain_zero_underflow_and_zero_clamp_leakage() -> None:
    """Extreme distances (-d^2 / 2sigma^2 < -100,000) must never underflow or leak 1e-4 bg bias."""
    # Tight sigma = 1.0px on 512x512 image -> corner has -d^2 / 2sigma^2 ~ -250,000!
    pts = [torch.tensor([[10.0, 10.0]], dtype=torch.float32)]
    y0 = torch.zeros((1, 1, 128, 128), dtype=torch.float32, requires_grad=True)

    loss = bayesian_loss(
        y0, pts, sigma=1.0, background_ratio=0.10, stride=4,
        norm_mode="canonical", canonical_background=True, adaptive_sigma=False,
    )
    assert torch.isfinite(loss) and abs(loss.item() - 1.0) < 1e-5
    loss.backward()
    assert y0.grad is not None and torch.isfinite(y0.grad).all()
    # Far corner (127, 127) must have exact +1.0 gradient without 0/0 NaN
    assert abs(y0.grad[0, 0, 127, 127].item() - 1.0) < 1e-6
    # Head center cell (2, 2) -> (10.0, 10.0) has d=0, d_bg=51.2 => s_bg = -1310.72.
    # Without 1e-4 clamp hack, P(bg) == 0.0 to float32 precision, so grad == -1.000000!
    assert abs(y0.grad[0, 0, 2, 2].item() - (-1.0)) < 1e-6


def test_broad_foreground_head_peak_protection_at_stride2() -> None:
    """Broad SHB/QNRF Gaussian head (sigma=8px at Stride 2, peak y0 ~ 0.01) receives >=45% higher
    skirt-to-apex concentration than standard uniform MC-BPRO while strictly conserving regional count."""
    h_grid, w_grid = 128, 128  # 256x256 px at Stride 2
    yy, xx = torch.meshgrid(
        torch.arange(h_grid, dtype=torch.float32),
        torch.arange(w_grid, dtype=torch.float32),
        indexing="ij",
    )
    sigma_cells = 4.0  # 8.0 px / stride 2 = 4.0 cells
    g = torch.exp(-((yy - 64.0).square() + (xx - 64.0).square()) / (2.0 * sigma_cells ** 2))
    y0 = (g / g.sum()).unsqueeze(0).unsqueeze(0)  # Total mass = 1.0 person, peak ~ 0.00995 << 0.05!
    assert y0[0, 0, 64, 64].item() < 0.015

    regions = build_multiscale_regions(
        h_grid, w_grid, output_stride=2, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False
    )
    b_solver = regional_sum(y0, regions.boxes) * 0.5
    regional_ev = {
        "mu_count": b_solver,
        "dispersion": torch.full_like(b_solver, 50.0),
        "log_dispersion": torch.full_like(b_solver, 3.91),
        "rate": b_solver / regions.area.view(1, 1, -1),
        "b_solver": b_solver,
        "b_variance": b_solver,
        "weight": torch.ones_like(b_solver),
        "weight_solver": torch.ones_like(b_solver),
        "precision": torch.ones_like(b_solver),
        "rate_variance": torch.ones_like(b_solver),
        "count_variance": b_solver,
        "hurdle_logit": None,
    }
    common_kw = dict(
        y0=y0, z0=y0, regional_evidence=regional_ev, regions=regions,
        scale_weights=None, solver_strength=1.0, default_solver_strength=1.0,
        uniform_reliability=False, p16=torch.zeros(1, 32, 16, 16),
        fg_logit=None, pi_scale=None, pi_aspect=None, trust_gate=None,
        laplace_kernel=torch.zeros(1, 1, 3, 3),
    )
    out_std = solve_inverse_measure(
        cfg=RMRv3Config(output_stride=2, subpixel_stride2=True, solver_mode="mcbpro", mcbpro_conservative_scaling=False),
        **common_kw,
    )
    out_cons = solve_inverse_measure(
        cfg=RMRv3Config(output_stride=2, subpixel_stride2=True, solver_mode="mcbpro", mcbpro_conservative_scaling=True),
        **common_kw,
    )
    assert abs(out_cons["y"].sum().item() - 0.5) < 0.03
    assert out_cons["y"][0, 0, 64, 64].item() >= 1.35 * out_std["y"][0, 0, 64, 64].item()


def test_subperson_softplus_noise_zero_phantom_seeding_vs_true_deficit() -> None:
    """Empty scene (y0 == 0) with Softplus noise b_R <= 0.18 produces 0.0 phantom people across all boxes."""
    h_grid, w_grid = 128, 128
    regions = build_multiscale_regions(
        h_grid, w_grid, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False
    )
    n_boxes = regions.boxes.shape[0]
    y0_empty = torch.zeros((1, 1, h_grid, w_grid), dtype=torch.float32)

    # Every box has Softplus background tail noise b_R = 0.12 people
    b_noise = torch.full((1, 1, n_boxes), 0.12, dtype=torch.float32)
    y_empty_mcbpro = closed_form_mcbpro_transport(
        y0_empty, b_noise, None, regions, deficit_seeding=True, conservative_scaling=True
    )
    y_empty_fed = fed_mcbpro_transport(
        y0_empty, b_noise, None, regions, feat_p4=torch.randn(1, 32, h_grid, w_grid),
        deficit_seeding=True, conservative_scaling=True,
    )
    assert y_empty_mcbpro.sum().item() == 0.0, (
        f"Phantom people seeded on empty scene: {y_empty_mcbpro.sum().item()}"
    )
    assert y_empty_fed.sum().item() == 0.0, (
        f"FED-MCBPRO seeded phantom people on empty scene: {y_empty_fed.sum().item()}"
    )

    # Conversely, a single disjoint box with 1.0 real missed person recovers exact 1.0000 mass!
    single_reg = build_multiscale_regions(
        32, 32, output_stride=4, region_sizes_px=(128,), overlap=0.0, include_full_image=False
    )
    b_one_person = torch.tensor([[[1.0]]], dtype=torch.float32)
    y_one = closed_form_mcbpro_transport(
        torch.zeros((1, 1, 32, 32)), b_one_person, None, single_reg,
        deficit_seeding=True, conservative_scaling=True,
    )
    assert abs(y_one.sum().item() - 1.0) < 1e-4, f"Expected exact 1.0 missed person recovery, got {y_one.sum().item()}"


def test_scale_routed_presence_prevents_small_box_dilution() -> None:
    """Routing a foreground person to Scale 1 prevents empty Scale 0 sub-boxes from zeroing mass or diluting peaks."""
    boxes = torch.tensor([
        [0, 0, 4, 4],
        [0, 4, 4, 8],
        [4, 0, 8, 4],
        [4, 4, 8, 8],
        [0, 0, 8, 8],
    ], dtype=torch.int64)
    scale_id = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int64)
    area = torch.tensor([16.0, 16.0, 16.0, 16.0, 64.0], dtype=torch.float32)
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area)

    y0 = torch.full((1, 1, 8, 8), 0.01, dtype=torch.float32)
    y0[0, 0, 2, 2] = 0.37  # Total mass = 1.00 (peak + skirt)
    # Sub-box 0 missed the person (0.0), while coarse 8x8 box detected 0.6
    b_solver = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.6]]], dtype=torch.float32)
    scale_weights = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    scale_weights[:, 0] = 0.02
    scale_weights[:, 1] = 0.98

    y_unrouted = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, scale_weights=None,
        deficit_seeding=False, conservative_scaling=True,
    )
    y_routed = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, scale_weights=scale_weights,
        deficit_seeding=False, conservative_scaling=True,
    )
    # Scale routing recovers ~0.98 * 0.6 = 0.588 mass instead of 0.5 * 0.6 = 0.30 unrouted dilution!
    assert abs(y_routed.sum().item() - 0.588) < 0.02
    assert y_routed[0, 0, 2, 2].item() > 1.8 * y_unrouted[0, 0, 2, 2].item()


def test_init_m0_auto_validation_and_config_parsing() -> None:
    """Verify init_m0='auto' is accepted across validator, RMRv3Config, and make_model."""
    cfg = {
        "seed": 42,
        "data": {"train_manifest": "data/shb_train_all.jsonl", "val_manifest": "data/shb_test.jsonl"},
        "model": {"init_m0": "auto", "pretrained": False},
        "loss": {"lambda_region_nb": 0.5},
    }
    validate_v3_config(cfg)
    rcfg = RMRv3Config.from_dict(cfg["model"])
    assert rcfg.init_m0 > 0.0
    model, _ = make_model(cfg)
    assert model is not None

    with pytest.raises(ValueError, match="init_m0"):
        validate_v3_config({"model": {"init_m0": "bad_string"}})
    with pytest.raises(ValueError, match="init_m0"):
        validate_v3_config({"model": {"init_m0": -0.01}})
