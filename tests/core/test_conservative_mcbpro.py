"""Unit tests for Post-Adjoint Morphological Peak-Protected Bregman-Poisson Transport.

Guarantees the core mathematical invariants of Conservative MC-BPRO (mcbpro.py & fed_mcbpro.py):
1. Dirac Carrier Peak Fidelity under Regional Saturation:
   When regional box evidence underestimates a dense crowd cluster (0 < b_R < mu_R),
   verified local carrier Dirac peaks (morphological prominence > 0, apex_ratio -> 1)
   are protected (s_final -> 1.0), preventing destructive peak attenuation.
2. 100% False-Positive Suppression on Empty Background:
   When regional box evidence is zero (b_R = 0), m_presence = 0 -> p_peak = 0,
   so spurious carrier peaks on empty background are 100% suppressed (y_cons = 0.0).
3. Zero Per-Box Jensen Positive Bias:
   Two overlapping windows with symmetric over/under-estimation (e.g., s_R1 = 1.25, s_R2 = 0.75)
   average linearly via regional_adjoint BEFORE peak protection (s_raw = 1.00), eliminating
   the positive Jensen bias of pre-adjoint per-box ReLU(s_R - 1).
4. Zero Flat-Plateau / Gaussian-Skirt Leakage:
   A constant flat plateau (y0 == C everywhere) has morphological prominence y0 - y0_min == 0,
   so p_peak == 0 and the plateau is faithfully scaled down to match b_R.
5. Deficit Recovery:
   When regional evidence detects missing mass (b_R > mu_R), the deficit is positively
   injected into valleys and density is scaled up.
6. Isolated Person Multiscale Fidelity:
   On full 512x512 multi-box geometries (32, 64, 128px), isolated individuals are
   preserved under regional saturation via scale-invariant presence gating.
7. Batch Sample Isolation, Prime Grid Parity, and End-to-End Autograd.
"""
from __future__ import annotations

import torch
from rmr_core.operators.mcbpro import closed_form_mcbpro_transport, regional_sum
from rmr_core.operators.fed_mcbpro import fed_mcbpro_transport
from rmr_core.operators.regions import RegionSet, build_multiscale_regions
from rmr_v3.model import RMRv3, RMRv3Config


def test_conservative_scaling_preserves_carrier_peaks_when_evidence_underestimates() -> None:
    """Invariant 1: Verified carrier peaks are protected when regional evidence saturates."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 3, 3] = 1.0
    mu_0 = y0.sum().item()

    # Under-estimated box evidence (0.8 < 2.0) in box 0 covering [0:8, 0:8]
    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 0.8

    # Standard solver destructively squashes peaks
    y_std = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=False)
    assert y_std[0, 0, 2, 2].item() < 0.5, "Standard solver should have attenuated peak"

    # Post-Adjoint Peak-Protected solver protects sharp local maxima
    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons[0, 0, 2, 2].item() >= 0.99, f"Carrier peak squashed: {y_cons[0, 0, 2, 2].item()}"
    assert y_cons[0, 0, 3, 3].item() >= 0.99, f"Carrier peak squashed: {y_cons[0, 0, 3, 3].item()}"
    assert y_cons.sum().item() >= mu_0 * 0.99, "Total foreground peak mass must be preserved"


def test_conservative_scaling_suppresses_false_positives_when_evidence_is_zero() -> None:
    """Invariant 2: Spurious carrier peaks on empty background (b_R = 0) are 100% suppressed."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    # Spurious carrier peak on empty background
    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0

    # Regional evidence confirms empty background (b_R = 0 everywhere)
    b_solver = torch.zeros(1, 1, n_boxes)

    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons.sum().item() < 1e-4, (
        f"False positive on empty background was not suppressed: {y_cons.sum().item()}"
    )


def test_conservative_scaling_eliminates_per_box_jensen_bias() -> None:
    """Invariant 3: Overlapping boxes with s_R1=1.25 and s_R2=0.75 average linearly to 1.00."""
    # Two identical overlapping boxes covering the full 8x8 grid
    boxes = torch.tensor([[0, 0, 8, 8], [0, 0, 8, 8]], dtype=torch.int64)
    scale_id = torch.tensor([0, 0], dtype=torch.int64)
    area = torch.tensor([64.0, 64.0], dtype=torch.float32)
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area)

    y0 = torch.zeros(1, 1, 8, 8)
    y0[0, 0, 4, 4] = 1.0

    # Box 0 overestimates (1.25), Box 1 underestimates (0.75) -> linear average is exactly 1.00
    b_solver = torch.tensor([[[1.25, 0.75]]], dtype=torch.float32)

    y_cons = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=True
    )
    # Pre-adjoint ReLU(s_R - 1) would have produced 1.0 + 0.5*(0.25 + 0) = 1.125 (Jensen bias!).
    # Post-adjoint linear averaging produces exactly 1.000.
    assert abs(y_cons[0, 0, 4, 4].item() - 1.0) < 1e-4, (
        f"Jensen positive bias detected: {y_cons[0, 0, 4, 4].item()} != 1.0"
    )


def test_conservative_scaling_scales_down_flat_plateau_without_leakage() -> None:
    """Invariant 4: Flat plateau (prominence == 0) receives zero peak protection and scales down."""
    boxes = torch.tensor([[0, 0, 8, 8]], dtype=torch.int64)
    scale_id = torch.tensor([0], dtype=torch.int64)
    area = torch.tensor([64.0], dtype=torch.float32)
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area)

    # Flat constant plateau: y0 == 0.1 everywhere -> sum = 6.4
    y0 = torch.full((1, 1, 8, 8), 0.1)
    # Regional evidence says total mass is 3.2 (half of 6.4)
    b_solver = torch.tensor([[[3.2]]], dtype=torch.float32)

    y_cons = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=True
    )
    # Because y0 is completely flat (y0 - y0_local_min == 0), p_peak == 0 everywhere,
    # so y_cons scales down cleanly to 3.2!
    assert abs(y_cons.sum().item() - 3.2) < 1e-2, (
        f"Flat plateau was not scaled down to match evidence: {y_cons.sum().item()} vs 3.2"
    )


def test_conservative_scaling_injects_deficit_when_evidence_detects_missing_mass() -> None:
    """Invariant 5: When evidence detects missing mass (deficit), mass must be scaled up."""
    regions = build_multiscale_regions(
        64, 64, output_stride=4, region_sizes_px=(32,), include_full_image=False
    )
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 3, 3] = 1.0
    mu_0 = y0.sum().item()

    # Deficit case: Evidence detects 4.0 people (carrier only has 2.0)
    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 4.0

    y_std = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=False)
    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons.sum().item() > mu_0 * 1.2, f"Deficit not scaled up: {y_cons.sum().item()} vs {mu_0}"
    assert y_cons[0, 0, 2, 2].item() > 1.0, "Existing peak should receive positive scaling boost"
    assert torch.allclose(y_cons, y_std, atol=1e-4), (
        "Pure deficit in Box 0 must match exact Bregman-Poisson without Jensen inflation"
    )


def test_conservative_scaling_isolated_person_multiscale_fidelity() -> None:
    """Invariant 6: Isolated individual on 512x512 multiscale boxes is preserved."""
    regions = build_multiscale_regions(512, 512, output_stride=2, region_sizes_px=(32, 64, 128))

    # Single isolated person
    y0 = torch.zeros(1, 1, 256, 256)
    y0[0, 0, 128, 128] = 1.0

    # Regional evidence underestimates by 50%
    b_solver = regional_sum(y0, regions.boxes.to(torch.int64)) * 0.5

    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons[0, 0, 128, 128].item() >= 0.96, (
        f"Isolated person squashed by multiscale solver: {y_cons[0, 0, 128, 128].item()}"
    )


def test_conservative_scaling_batch_sample_isolation() -> None:
    """Invariant 7: Batch sample 0 and sample 1 must have zero gradient cross-talk."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(2, 1, 16, 16, requires_grad=True)
    with torch.no_grad():
        y0[0, 0, 5, 5] = 2.0  # Sample 0 has crowd
        # Sample 1 is all zero

    b_solver = torch.zeros(2, 1, n_boxes, requires_grad=True)
    with torch.no_grad():
        b_solver[0, 0, 0] = 3.0  # Sample 0 has evidence

    y_out = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)

    loss_sample0 = y_out[0].sum()
    loss_sample0.backward()

    assert y0.grad is not None
    assert (y0.grad[1] == 0.0).all(), "Gradient cross-talk: Sample 0 loss influenced Sample 1!"
    assert (b_solver.grad[1] == 0.0).all(), "Evidence gradient cross-talk!"


def test_conservative_scaling_prime_dimensions() -> None:
    """Invariant 8: Arbitrary prime grid dimensions (e.g. 17 x 23 cells)."""
    h_c, w_c = 17, 23
    regions = build_multiscale_regions(h_c * 4, w_c * 4, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.rand(1, 1, h_c, w_c)
    b_solver = torch.rand(1, 1, n_boxes)

    y_out = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_out.shape == (1, 1, h_c, w_c)
    assert (y_out >= 0.0).all()
    assert torch.isfinite(y_out).all()


def test_fed_mcbpro_conservative_scaling_parity() -> None:
    """Invariant 9: FED-MCBPRO conservative scaling parity (both peak protection and FP suppression)."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 6, 6] = 1.0

    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 0.8
    feat_p4 = torch.randn(1, 48, 16, 16)

    y_fed_cons = fed_mcbpro_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=None,
        regions=regions,
        feat_p4=feat_p4,
        conservative_scaling=True,
    )
    assert y_fed_cons[0, 0, 2, 2].item() >= 0.99, "FED-MCBPRO carrier peak must be preserved"

    # Also verify empty background suppression in FED-MCBPRO
    b_zero = torch.zeros(1, 1, n_boxes)
    y_fed_zero = fed_mcbpro_transport(
        y0=y0,
        b_solver=b_zero,
        weight_solver=None,
        regions=regions,
        feat_p4=feat_p4,
        conservative_scaling=True,
    )
    assert y_fed_zero.sum().item() < 1e-4, "FED-MCBPRO must suppress false positives when b_R = 0"


def test_rmrv3_model_pipeline_with_conservative_scaling() -> None:
    """Invariant 10: Full RMRv3 forward-backward integration with conservative scaling."""
    cfg = RMRv3Config(
        feature_width=32,
        subpixel_stride2=False,
        output_stride=4,
        enable_solver=True,
        solver_mode="mcbpro",
        mcbpro_conservative_scaling=True,
        floor_tau=0.003,
        pretrained=False,
    )
    model = RMRv3(cfg)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    out = model(x)

    assert "y" in out and "y0" in out
    loss = out["y"].sum()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))


def test_spatially_stratified_crowd_aware_crop_balances_sparse_and_dense_regions() -> None:
    """Invariant 11: Spatially stratified 64x64 cell-balanced crowd-aware crop prevents
    dense horizon clusters from starving sparse foreground heads."""
    import random
    import numpy as np
    from PIL import Image
    from rmr_core.transforms import train_transform

    random.seed(42)
    img = Image.fromarray(np.full((1024, 1024, 3), 128, dtype=np.uint8))

    # 200 dense points packed inside top-left 32x32 cell [10..20, 10..20]
    dense_pts = torch.full((200, 2), 15.0)
    # 1 sparse foreground point in bottom-right cell [900, 900] (disjoint 256x256 crop)
    sparse_pt = torch.tensor([[900.0, 900.0]])
    pts = torch.cat([dense_pts, sparse_pt], dim=0)

    sparse_hits = 0
    trials = 100
    for _ in range(trials):
        _, c_pts = train_transform(
            img,
            pts,
            crop_size=256,
            scale_range=(1.0, 1.0),
            hflip_prob=0.0,
            crowd_aware_crop=True,
            crowd_aware_prob=1.0,
        )
        # If the sparse foreground point was selected as anchor, the crop has exactly 1 point
        if c_pts.shape[0] == 1:
            sparse_hits += 1

    # Naive point-index sampling would hit the sparse foreground point ~0.5% of the time (0-2 hits).
    # Stratified 64x64 cell sampling gives each of the 2 occupied cells 50% probability (~50 hits).
    assert 30 <= sparse_hits <= 70, (
        f"Expected ~50% stratified cell balance for sparse foreground, got {sparse_hits}/{trials}"
    )

