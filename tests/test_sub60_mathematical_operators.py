from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    weighted_normalized_adjoint_field,
)
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.regional_head import apply_scale_consistency_gating
from rmr_v3.model import RMRv3, RMRv3Config


def test_shifted_carrier_escapes_zero_absorbing_trap():
    """Verify that Shifted-Carrier Radon-Nikodym breaks the zero-absorbing trap

    without causing mass leakage to empty background.
    """
    device = torch.device("cpu")
    h, w = 32, 32
    # Define a single 16x16 region at center
    boxes = torch.tensor([[8, 8, 24, 24]], dtype=torch.int64)
    area = torch.tensor([256], dtype=torch.int64)
    scale_id = torch.tensor([0], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, area=area, scale_id=scale_id)

    # Regional ground truth observation: b = 10.0 (crowd present)
    b_solver = torch.tensor([[10.0]])
    weight_solver = torch.tensor([[1.0]])

    # Current state y has been wiped to 0.0 at the center pixel (16, 16)
    y_curr = torch.zeros((1, 1, h, w), dtype=torch.float32)
    # But neural carrier y0 had detected crowd evidence at (16, 16)
    y0 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    y0[0, 0, 16, 16] = 0.5  # Positive carrier signal
    # Background pixel (0, 0) has 0.0 in both y and y0
    assert y_curr[0, 0, 0, 0] == 0.0
    assert y0[0, 0, 0, 0] == 0.0

    # 1. Standard Radon-Nikodym (shifted_carrier=False):
    field_std = weighted_normalized_adjoint_field(
        y_curr,
        b_solver,
        weight_solver,
        regions,
        adjoint_mode="radon_nikodym",
        shifted_carrier=False,
    )
    # Standard RN produces ZERO update at (16, 16) because y_curr[16, 16] == 0 (TRAPPED!)
    assert field_std[0, 0, 16, 16].item() == 0.0

    # 2. Shifted-Carrier Radon-Nikodym (shifted_carrier=True):
    field_shifted = weighted_normalized_adjoint_field(
        y_curr,
        b_solver,
        weight_solver,
        regions,
        adjoint_mode="radon_nikodym",
        shifted_carrier=True,
        shifted_carrier_eps=0.05,
        y_initial=y0,
    )
    # Shifted RN produces a NON-ZERO (negative residual) update at (16, 16)!
    # Negative field means step_delta = omega * field < 0 => y_next = y - step_delta > 0 (mass added!)
    assert field_shifted[0, 0, 16, 16].item() < -1e-4

    # 3. ZERO Background Leakage Invariant:
    # At pure background pixel (0, 0) where y0 == 0 and y_curr == 0, field must remain EXACTLY 0.0!
    assert field_shifted[0, 0, 0, 0].item() == 0.0


def test_density_adaptive_trust_region():
    """Verify that density-adaptive trust region expands kappa in dense areas

    while preserving standard kappa in sparse areas.
    """
    device = torch.device("cpu")
    h, w = 16, 16
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(8,), device=device)
    b_solver = torch.full((1, 1, len(regions.boxes)), 5.0)
    weight_solver = torch.ones_like(b_solver)

    # Initial state with sparse background and a dense clump
    y0 = torch.full((1, 1, h, w), 0.01)
    y0[0, 0, 4:8, 4:8] = 0.50  # Dense clump

    # Run solver with density_adaptive_trust=False
    res_fixed = unrolled_sirt_solver(
        y0, b_solver, weight_solver, regions,
        iterations=1, trust_region_kappa=0.35,
        density_adaptive_trust=False,
    )
    # Run solver with density_adaptive_trust=True
    res_adapt = unrolled_sirt_solver(
        y0, b_solver, weight_solver, regions,
        iterations=1, trust_region_kappa=0.35,
        density_adaptive_trust=True,
        trust_dense_tau=0.10,
        trust_dense_kappa=0.80,
    )

    y_fixed = res_fixed["y"]
    y_adapt = res_adapt["y"]

    # In the dense clump, adaptive trust allows greater mass inflow towards target 5.0
    assert y_adapt[0, 0, 6, 6].item() >= y_fixed[0, 0, 6, 6].item()
    # All values must remain finite and non-negative
    assert torch.isfinite(y_adapt).all()
    assert (y_adapt >= 0.0).all()


def test_density_scale_gating():
    """Verify that density_scale_gating smoothly suppresses large boxes

    (scale k=2, e.g. 128px) when regional rate is high.
    """
    device = torch.device("cpu")
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(8, 16, 32), device=device)
    scale_weights = torch.full((1, 3, h, w), 1.0 / 3.0)
    weights = torch.ones((1, 1, len(regions.boxes)))

    # Sparse rates: rate < 0.15
    rate_sparse = torch.full((1, len(regions.boxes)), 0.02)
    w_sparse = apply_scale_consistency_gating(
        weights, regions, scale_weights,
        regional_rate=rate_sparse,
        density_scale_gating=True,
        density_scale_tau=0.15,
    )

    # Dense rates: rate > 0.15
    rate_dense = torch.full((1, len(regions.boxes)), 0.35)
    w_dense = apply_scale_consistency_gating(
        weights, regions, scale_weights,
        regional_rate=rate_dense,
        density_scale_gating=True,
        density_scale_tau=0.15,
    )

    # For the largest scale (k=2), dense rate must result in lower weight than sparse rate
    mask_large = (regions.scale_id == 2)
    assert w_dense[0, 0, mask_large].mean().item() < w_sparse[0, 0, mask_large].mean().item()


def test_rmrv3_forward_with_all_sub60_operators():
    """Verify full end-to-end forward pass with all 3 operators active

    and parameter count strictly equals 104,441.
    """
    from rmr_v3.config import load_config
    raw_cfg = load_config("configs/rmr_sub60/sub60_canonical_reconciliation.yaml")["model"]
    raw_cfg["shifted_carrier"] = True
    raw_cfg["shifted_carrier_eps"] = 0.02
    raw_cfg["density_adaptive_trust"] = True
    raw_cfg["trust_dense_tau"] = 0.10
    raw_cfg["trust_dense_kappa"] = 0.80
    raw_cfg["pre_solver_scale_gating"] = True
    raw_cfg["density_scale_gating"] = True
    raw_cfg["density_scale_tau"] = 0.15
    cfg = RMRv3Config(**raw_cfg)
    device = torch.device("cpu")
    model = RMRv3(cfg).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 104441, f"Expected 104,441 parameters, got {total_params}"

    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    assert torch.isfinite(out["y"]).all()
    assert (out["y"] >= 0.0).all()


def test_sub60_operator_validation_and_bypass_guards():
    """Verify that validator catches negative/invalid parameters and prevents silent bypass."""
    from rmr_v3.config import load_config, validate_v3_config
    import pytest

    # Invalid shifted_carrier_eps
    c1 = load_config("configs/rmr_sub60/sub60_canonical_reconciliation.yaml")
    c1["model"]["shifted_carrier_eps"] = -0.05
    with pytest.raises(ValueError, match="shifted_carrier_eps"):
        validate_v3_config(c1)

    # Invalid trust_dense_kappa
    c2 = load_config("configs/rmr_sub60/sub60_canonical_reconciliation.yaml")
    c2["model"]["trust_dense_kappa"] = 0.0
    with pytest.raises(ValueError, match="trust_dense_kappa"):
        validate_v3_config(c2)

    # Density scale gating requires dynamic scale routing
    c3 = load_config("configs/rmr_sub60/sub60_canonical_reconciliation.yaml")
    c3["model"]["density_scale_gating"] = True
    c3["model"]["dynamic_scale_routing"] = False
    with pytest.raises(ValueError, match="dynamic_scale_routing"):
        validate_v3_config(c3)

