from __future__ import annotations

import torch
import pytest

from rmr_core.operators import RegionSet
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.solver_trust import apply_trust_region_clamping


def test_zero_carrier_deficit_recovery_uncapped():
    """Verify that zero-support cells with positive deficit can add mass and are not strangled to 0.00175."""
    device = torch.device("cpu")
    h, w = 16, 16
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.int64)
    area = torch.tensor([256], dtype=torch.int64)
    scale_id = torch.tensor([0], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, area=area, scale_id=scale_id)

    # Completely zero initial carrier
    y0 = torch.zeros((1, 1, h, w), device=device)
    # Evidence demands 50.0 people in 256 cells => 50/256 ≈ 0.1953 people/cell
    b_solver = torch.tensor([[50.0]], device=device)
    w_solver = torch.tensor([[1.0]], device=device)

    res = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_solver,
        weight_solver=w_solver,
        regions=regions,
        iterations=1,
        omega=1.0,
        trust_region_kappa=0.35,
        density_adaptive_trust=False,
    )
    y1 = res["y"]
    max_increase = (y1 - y0).max().item()

    # Old buggy behavior strangled max_increase to 0.35 * 0.005 = 0.00175
    # Decoupled behavior allows true deficit step ~0.1953
    assert max_increase > 0.15, (
        f"Zero-cell starvation detected! Expected max_increase > 0.15, got {max_increase}"
    )
    assert max_increase <= 0.35 + 1e-4, (
        f"Trust region cap exceeded! Expected max_increase <= 0.35, got {max_increase}"
    )


def test_zero_carrier_mass_removal_strictly_bounded():
    """Verify that empty cells never remove mass (bound_removal is strictly bounded by eff_floor)."""
    delta_y = torch.tensor([-5.0, 5.0])
    z_state = torch.tensor([0.0, 1.0])

    # eff_kappa = 0.35, eff_floor = 0.005
    clamped = apply_trust_region_clamping(
        delta_y=delta_y,
        z_state=z_state,
        trust_region_kappa=0.35,
        effective_trust_floor=0.005,
        area_scale=1.0,
    )

    # Empty cell (z=0) removal must be clamped to -0.35 * 0.005 = -0.00175 (non-negative protection)
    assert clamped[0].item() >= -0.00175 - 1e-6
    # Non-empty cell (z=1) addition must be clamped to 0.35 * 1.0 = 0.35
    assert clamped[1].item() <= 0.35 + 1e-6
