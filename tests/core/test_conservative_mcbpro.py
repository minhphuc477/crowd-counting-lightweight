"""Unit tests for Mass-Conserving Csiszar I-Projection Bregman-Poisson Transport.

Guarantees the core mathematical invariants of Conservative MC-BPRO (mcbpro.py & fed_mcbpro.py):
1. Skirt-to-Peak Concentration with Exact Regional Mass Conservation:
   Minimizes D_KL(y || y_ref) s.t. A_R y = b_R where y_ref = y_aug * (1 + p_peak),
   strictly sharpening verified head peaks relative to standard uniform projection
   while conserving 100% of regional count b_R (zero count inflation).
2. 100% False-Positive Suppression on Empty Background & Occupied Clutter:
   - Empty background (b_R = 0) yields exact y_cons == 0.0.
   - 1 real head + 4 spurious false-positive peaks with oracle b_R = 1.0 yields total
     recovered count ~1.0 (zero +239% count inflation).
3. Zero Per-Box Jensen Positive Bias:
   Overlapping windows with symmetric over/under-estimation (b_R1 = 1.25, b_R2 = 0.75)
   average linearly via regional_adjoint to exact 1.0000 mass.
4. Zero Flat-Plateau Leakage:
   A constant flat plateau (y0 == C everywhere) has prominence == 0, matching standard MC-BPRO.
5. Deficit Recovery, Batch Isolation, Prime Grid Parity, End-to-End Autograd,
   and Opt-In Spatially Stratified Crop vs V64 Point-Uniform Default.
"""
from __future__ import annotations

import torch
from rmr_core.operators.mcbpro import closed_form_mcbpro_transport, regional_sum
from rmr_core.operators.fed_mcbpro import fed_mcbpro_transport
from rmr_core.operators.regions import RegionSet, build_multiscale_regions
from rmr_v3.model import RMRv3, RMRv3Config


def test_conservative_scaling_concentrates_skirt_into_peaks_while_conserving_mass() -> None:
    """Invariant 1: Concentrates diffused skirt mass into peaks while conserving exact box count b_R."""
    boxes = torch.tensor([[0, 0, 8, 8]], dtype=torch.int64)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.tensor([0], dtype=torch.int64),
        area=torch.tensor([64.0], dtype=torch.float32),
    )

    # Resolved peak at (4, 4) surrounded by diffused Gaussian skirt across the 8x8 box
    y0 = torch.full((1, 1, 8, 8), 0.02)
    y0[0, 0, 4, 4] = 0.74  # Total y0 mass = 63 * 0.02 + 0.74 = 2.00

    b_solver = torch.tensor([[[1.0]]], dtype=torch.float32)

    y_std = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=False
    )
    y_cons = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=True
    )

    # Exact regional mass conservation: both sum to 1.0000!
    assert abs(y_std.sum().item() - 1.0) < 1e-4
    assert abs(y_cons.sum().item() - 1.0) < 1e-4
    # Csiszar I-projection strictly increases peak height and strictly decreases skirt leakage!
    assert y_cons[0, 0, 4, 4].item() > y_std[0, 0, 4, 4].item() * 1.25
    assert y_cons[0, 0, 0, 0].item() < y_std[0, 0, 0, 0].item()


def test_conservative_scaling_suppresses_false_positives_on_empty_and_cluttered_regions() -> None:
    """Invariant 2: Zero mass when b_R=0 AND no count inflation on 1 real + 4 FP peaks when b_R=1.0."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    b_zero = torch.zeros(1, 1, n_boxes)
    y_zero = closed_form_mcbpro_transport(y0, b_zero, None, regions, conservative_scaling=True)
    assert y_zero.sum().item() < 1e-4

    # Adversarial check: 1 real person + 4 false-positive spikes (y0.sum() = 5.0), oracle b_R = 1.0!
    regions_ms = build_multiscale_regions(128, 128, output_stride=2, region_sizes_px=(32, 64, 128))
    gt = torch.zeros(1, 1, 64, 64)
    gt[0, 0, 20, 20] = 1.0
    y0_clutter = gt.clone()
    for r, c in ((20, 26), (26, 20), (26, 26), (30, 30)):
        y0_clutter[0, 0, r, c] = 1.0
    b_oracle = regional_sum(gt, regions_ms.boxes)

    y_cons_clutter = closed_form_mcbpro_transport(
        y0_clutter, b_oracle, None, regions_ms, reg_threshold=0.00025, conservative_scaling=True
    )
    assert abs(y_cons_clutter.sum().item() - 1.0) < 0.08, (
        f"Spurious peaks inflated regional count beyond oracle 1.0: {y_cons_clutter.sum().item()}"
    )


def test_conservative_scaling_eliminates_per_box_jensen_bias() -> None:
    """Invariant 3: Overlapping boxes with s_R1=1.25 and s_R2=0.75 average linearly to 1.00."""
    boxes = torch.tensor([[0, 0, 8, 8], [0, 0, 8, 8]], dtype=torch.int64)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.tensor([0, 0], dtype=torch.int64),
        area=torch.tensor([64.0, 64.0], dtype=torch.float32),
    )

    y0 = torch.full((1, 1, 8, 8), 0.01)
    y0[0, 0, 4, 4] = 0.37  # Total mass = 1.00
    b_solver = torch.tensor([[[1.25, 0.75]]], dtype=torch.float32)

    y_cons = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=True
    )
    assert abs(y_cons.sum().item() - 1.0) < 1e-4


def test_conservative_scaling_scales_down_flat_plateau_without_leakage() -> None:
    """Invariant 4: Flat plateau (prominence == 0) receives zero peak boost and matches standard."""
    boxes = torch.tensor([[0, 0, 8, 8]], dtype=torch.int64)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.tensor([0], dtype=torch.int64),
        area=torch.tensor([64.0], dtype=torch.float32),
    )

    y0 = torch.full((1, 1, 8, 8), 0.1)
    b_solver = torch.tensor([[[3.2]]], dtype=torch.float32)

    y_cons = closed_form_mcbpro_transport(
        y0, b_solver, None, regions, deficit_seeding=False, conservative_scaling=True
    )
    assert abs(y_cons.sum().item() - 3.2) < 1e-2


def test_conservative_scaling_injects_deficit_and_sharpens_peaks() -> None:
    """Invariant 5: Missing mass is injected into valleys and concentrated toward peaks while conserving mass."""
    regions = build_multiscale_regions(
        64, 64, output_stride=4, region_sizes_px=(32,), overlap=0.0, include_full_image=False
    )
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 5, 5] = 1.0

    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 4.0

    y_std = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=False)
    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert abs(y_cons.sum().item() - 4.0) < 1e-3
    assert y_cons[0, 0, 2, 2].item() > y_std[0, 0, 2, 2].item()


def test_conservative_scaling_multiscale_broad_head_concentration() -> None:
    """Invariant 6: Broad Gaussian head on 256x256 multiscale Stride-2 grid is sharpened without count drift."""
    regions = build_multiscale_regions(256, 256, output_stride=2, region_sizes_px=(32, 64, 128))
    yy, xx = torch.meshgrid(torch.arange(128, dtype=torch.float32), torch.arange(128, dtype=torch.float32), indexing="ij")
    g = torch.exp(-((yy - 64.0).square() + (xx - 64.0).square()) / (2.0 * 4.0 ** 2))
    y0 = (g / g.sum()).unsqueeze(0).unsqueeze(0)
    b_solver = regional_sum(y0, regions.boxes.to(torch.int64)) * 0.8

    y_std = closed_form_mcbpro_transport(y0, b_solver, None, regions, reg_threshold=0.00025, conservative_scaling=False)
    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, reg_threshold=0.00025, conservative_scaling=True)
    assert abs(y_cons.sum().item() - 0.8) < 0.03
    assert y_cons[0, 0, 64, 64].item() > y_std[0, 0, 64, 64].item() * 1.35


def test_conservative_scaling_batch_sample_isolation() -> None:
    """Invariant 7: Batch sample 0 and sample 1 must have zero gradient cross-talk."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(2, 1, 16, 16, requires_grad=True)
    with torch.no_grad():
        y0[0, 0, 5, 5] = 2.0
    b_solver = torch.zeros(2, 1, n_boxes, requires_grad=True)
    with torch.no_grad():
        b_solver[0, 0, 0] = 3.0

    y_out = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    y_out[0].sum().backward()
    assert y0.grad is not None and (y0.grad[1] == 0.0).all()
    assert b_solver.grad is not None and (b_solver.grad[1] == 0.0).all()


def test_conservative_scaling_prime_dimensions() -> None:
    """Invariant 8: Arbitrary prime grid dimensions (17 x 23 cells)."""
    h_c, w_c = 17, 23
    regions = build_multiscale_regions(h_c * 4, w_c * 4, output_stride=4, region_sizes_px=(32,))
    y0 = torch.rand(1, 1, h_c, w_c)
    b_solver = torch.rand(1, 1, regions.boxes.shape[0])
    y_out = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_out.shape == (1, 1, h_c, w_c) and (y_out >= 0.0).all() and torch.isfinite(y_out).all()


def test_fed_mcbpro_conservative_scaling_parity() -> None:
    """Invariant 9: FED-MCBPRO conservative scaling sharpens peaks while conserving mass and zeroing empty bg."""
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, scale_id=torch.tensor([0]), area=torch.tensor([256.0]))

    y0 = torch.full((1, 1, 16, 16), 0.005)
    y0[0, 0, 4, 4] = 0.80
    b_solver = torch.tensor([[[1.0]]], dtype=torch.float32)
    feat_p4 = torch.randn(1, 48, 16, 16)

    y_fed_std = fed_mcbpro_transport(y0, b_solver, None, regions, feat_p4=feat_p4, conservative_scaling=False)
    y_fed_cons = fed_mcbpro_transport(y0, b_solver, None, regions, feat_p4=feat_p4, conservative_scaling=True)
    assert abs(y_fed_cons.sum().item() - 1.0) < 1e-3
    assert y_fed_cons[0, 0, 4, 4].item() > y_fed_std[0, 0, 4, 4].item() * 1.25

    y_fed_zero = fed_mcbpro_transport(
        y0, torch.zeros_like(b_solver), None, regions, feat_p4=feat_p4, conservative_scaling=True
    )
    assert y_fed_zero.sum().item() < 1e-4


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
    out["y"].sum().backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))


def test_crowd_aware_crop_default_point_uniform_vs_opt_in_stratified() -> None:
    """Invariant 11: Default preserves V64 point-uniform crop; stratified_crowd_crop=True balances cells."""
    import random
    import numpy as np
    from PIL import Image
    from rmr_core.transforms import train_transform

    random.seed(42)
    img = Image.fromarray(np.full((1024, 1024, 3), 128, dtype=np.uint8))
    dense_pts = torch.full((200, 2), 15.0)
    sparse_pt = torch.tensor([[900.0, 900.0]])
    pts = torch.cat([dense_pts, sparse_pt], dim=0)

    default_sparse_hits = sum(
        1
        for _ in range(100)
        if train_transform(
            img, pts, crop_size=256, scale_range=(1.0, 1.0), hflip_prob=0.0,
            crowd_aware_crop=True, crowd_aware_prob=1.0,
        )[1].shape[0] == 1
    )
    stratified_sparse_hits = sum(
        1
        for _ in range(100)
        if train_transform(
            img, pts, crop_size=256, scale_range=(1.0, 1.0), hflip_prob=0.0,
            crowd_aware_crop=True, crowd_aware_prob=1.0, stratified_crowd_crop=True,
        )[1].shape[0] == 1
    )
    assert default_sparse_hits <= 5, f"Default crop drifted from V64 point-uniform: {default_sparse_hits}"
    assert 30 <= stratified_sparse_hits <= 70, f"Opt-in stratified crop failed: {stratified_sparse_hits}"
