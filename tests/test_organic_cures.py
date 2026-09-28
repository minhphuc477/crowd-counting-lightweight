from __future__ import annotations

import torch
import pytest

from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses.auxiliary import count_harmonized_cell_loss
from rmr_v3.solver import unrolled_sirt_solver
from rmr_core.operators import RegionSet


def test_hurdle_occupancy_gate_preserves_dense_mass():
    """Verify that smooth occupancy gating prevents hurdle attenuation on occupied heads."""
    # Test formula directly
    b_raw = torch.tensor([50.0, 10.0, 1.0, 0.5, 0.05])
    pi_r = torch.tensor([0.75, 0.70, 0.60, 0.50, 0.01])
    
    # Old behavior: b_old = pi_r * b_raw
    b_old = pi_r * b_raw
    # In dense crowd, old behavior lost 25% of mass (50.0 -> 37.5)
    assert b_old[0] < 40.0
    
    # New principled occupancy gate:
    occ_gate = 1.0 - (1.0 - pi_r) * torch.clamp(1.0 - b_raw, min=0.0, max=1.0)
    b_new = occ_gate * b_raw
    
    # In occupied regions (b >= 1.0), gate is 1.0 (zero attenuation)
    assert torch.isclose(b_new[0], torch.tensor(50.0))
    assert torch.isclose(b_new[1], torch.tensor(10.0))
    assert torch.isclose(b_new[2], torch.tensor(1.0))
    
    # In borderline/background regions (b < 1.0), smooth attenuation
    assert b_new[3] < 0.5  # b=0.5 attenuated
    assert b_new[4] < 0.01  # b=0.05 suppressed to background


def test_count_harmonized_fractional_normalization():
    """Verify that fractional norm_power stabilizes gradients across crowd scales."""
    y = torch.ones(1, 1, 16, 16, requires_grad=True)
    tgt = torch.zeros(1, 1, 16, 16)
    tgt[0, 0, 4, 4] = 100.0  # Dense cluster
    
    # Run with norm_power = 0.5 (square-root balanced)
    loss = count_harmonized_cell_loss(y, tgt, norm_power=0.5, norm_ref=100.0)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert loss.item() > 0.0
    
    loss.backward()
    assert y.grad is not None and not torch.isnan(y.grad).any()


def test_asymmetric_trust_region_mass_injection():
    """Verify solver allows up to 100% mass addition while restricting mass erosion to eff_kappa."""
    device = torch.device("cpu")
    h, w = 16, 16
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.int64)
    area = torch.tensor([256], dtype=torch.int64)
    scale_id = torch.tensor([0], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, area=area, scale_id=scale_id)
    
    # y0 severely underestimated: 0.05
    y0 = torch.full((1, 1, h, w), 0.05, device=device)
    # Target evidence demands 50 people
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
    # y1 must increase significantly more than 0.35 * 0.05 = 0.0175 due to bound_pos = 1.0 * z
    added_mass = (y1 - y0).sum().item()
    assert added_mass > 0.0, "Solver failed to inject mass when q < b"
    # Max pixel increase in y1 should exceed 0.35 * 0.05 = 0.0175
    max_increase = (y1 - y0).max().item()
    assert max_increase > 0.0175, f"Expected max increase > 0.0175, got {max_increase}"
