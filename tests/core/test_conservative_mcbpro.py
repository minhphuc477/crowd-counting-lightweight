"""Unit tests for Conservative (Non-Destructive) Bregman-Poisson Measure Recovery.

Guarantees the core mathematical invariants of Conservative MC-BPRO:
1. Dirac Carrier Fidelity: Carrier peaks are strictly preserved (y* >= y0 on foreground).
   When regional box evidence underestimates the count due to spatial box smoothing,
   the solver never squashes down sharp carrier Dirac peaks.
2. Deficit Recovery: When regional evidence detects missing mass (b_R > mu_R),
   the deficit is positively injected into valleys and density is scaled up.
3. Isolated Person Multiscale Fidelity: On full 512x512 multi-box geometries (32, 64, 128px),
   isolated individuals are 100% preserved even under under-estimated evidence.
4. Batch Sample Isolation: Zero cross-batch broadcasting or gradient leakage.
5. Arbitrary & Prime Resolution Parity: Finite, non-negative output on odd/prime grids.
6. Smooth Autograd: Full end-to-end differentiability with finite gradients.
"""
from __future__ import annotations

import torch
from rmr_core.operators.mcbpro import closed_form_mcbpro_transport, regional_sum
from rmr_core.operators.fed_mcbpro import fed_mcbpro_transport
from rmr_core.operators.regions import build_multiscale_regions
from rmr_v3.model import RMRv3, RMRv3Config


def test_conservative_scaling_preserves_carrier_peaks_when_evidence_underestimates() -> None:
    """Invariant 1: Solver must not squash carrier peaks when evidence underestimates."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 3, 3] = 1.0
    mu_0 = y0.sum().item()

    # Under-estimated box evidence (0.8 < 2.0)
    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 0.8

    # Standard solver (destructively squashes peaks)
    y_std = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=False)
    assert y_std[0, 0, 2, 2].item() < 0.5, "Standard solver should have attenuated peak"

    # Conservative solver (preserves carrier peaks)
    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons[0, 0, 2, 2].item() >= 0.999, f"Carrier peak squashed: {y_cons[0, 0, 2, 2].item()}"
    assert y_cons[0, 0, 3, 3].item() >= 0.999, f"Carrier peak squashed: {y_cons[0, 0, 3, 3].item()}"
    assert y_cons.sum().item() >= mu_0 * 0.999, "Total foreground mass must be preserved"


def test_conservative_scaling_injects_deficit_when_evidence_detects_missing_mass() -> None:
    """Invariant 2: When evidence detects missing mass (deficit), mass must be scaled up."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 3, 3] = 1.0
    mu_0 = y0.sum().item()

    # Deficit case: Evidence detects 4.0 people (carrier only has 2.0)
    b_solver = torch.zeros(1, 1, n_boxes)
    b_solver[0, 0, 0] = 4.0

    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons.sum().item() > mu_0 * 1.2, f"Deficit not scaled up: {y_cons.sum().item()} vs {mu_0}"
    assert y_cons[0, 0, 2, 2].item() > 1.0, "Existing peak should receive positive scaling boost"


def test_conservative_scaling_isolated_person_multiscale_fidelity() -> None:
    """Invariant 3: Isolated individual on 512x512 multiscale boxes is 100% preserved."""
    regions = build_multiscale_regions(512, 512, output_stride=2, region_sizes_px=(32, 64, 128))

    # Single isolated person
    y0 = torch.zeros(1, 1, 256, 256)
    y0[0, 0, 128, 128] = 1.0

    # Regional evidence underestimates
    b_solver = regional_sum(y0, regions.boxes.to(torch.int64)) * 0.5

    y_cons = closed_form_mcbpro_transport(y0, b_solver, None, regions, conservative_scaling=True)
    assert y_cons[0, 0, 128, 128].item() >= 0.999, (
        f"Isolated person squashed by multiscale solver: {y_cons[0, 0, 128, 128].item()}"
    )
    assert y_cons.sum().item() >= 0.999, f"Mass lost: {y_cons.sum().item()}"


def test_conservative_scaling_batch_sample_isolation() -> None:
    """Invariant 4: Batch sample 0 and sample 1 must have zero gradient cross-talk."""
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

    # Compute loss on Sample 0 ONLY
    loss_sample0 = y_out[0].sum()
    loss_sample0.backward()

    # Sample 1 MUST receive identically zero gradient
    assert y0.grad is not None
    assert (y0.grad[1] == 0.0).all(), "Gradient cross-talk: Sample 0 loss influenced Sample 1!"
    assert (b_solver.grad[1] == 0.0).all(), "Evidence gradient cross-talk!"


def test_conservative_scaling_prime_dimensions() -> None:
    """Invariant 5: Arbitrary prime grid dimensions (e.g. 113 x 127 cells)."""
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
    """Invariant 6: FED-MCBPRO conservative scaling parity."""
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32,))
    n_boxes = regions.boxes.shape[0]

    y0 = torch.zeros(1, 1, 16, 16)
    y0[0, 0, 2, 2] = 1.0
    y0[0, 0, 3, 3] = 1.0

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
    assert y_fed_cons[0, 0, 2, 2].item() >= 0.999, "FED-MCBPRO carrier peak must be preserved"


def test_rmrv3_model_pipeline_with_conservative_scaling() -> None:
    """Invariant 7: Full RMRv3 forward-backward integration with conservative scaling."""
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
