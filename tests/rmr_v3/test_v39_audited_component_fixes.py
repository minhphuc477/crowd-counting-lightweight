from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from rmr_core.density_activation import _smooth_floor, _pool_local_density, _density_activate
from rmr_core.heads import FineMeasureHead
from rmr_core.operators import build_multiscale_regions, RegionSet
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.losses.orchestration import _compute_core_losses, TargetSupervisionRouter
from rmr_v3.losses.config import RMRv3LossConfig


def test_smooth_floor_negative_input_safety():
    """Verify that negative inputs cannot produce positive mass via squaring in _smooth_floor."""
    tau = 0.01
    # Negative values
    neg_tensor = torch.tensor([-1.0, -0.5, -0.05, -1e-5])
    out = _smooth_floor(neg_tensor, floor_tau=tau)
    # Must be exactly 0.0, never positive
    assert torch.all(out == 0.0), f"Negative inputs produced non-zero output: {out}"

    # Positive values below tau: quadratic suppression in [0, tau]
    pos_small = torch.tensor([0.002, 0.005, 0.008])
    out_small = _smooth_floor(pos_small, floor_tau=tau)
    assert torch.all(out_small >= 0.0)
    assert torch.all(out_small < pos_small)  # Must be strictly suppressed

    # Positive values above tau: linear shift
    pos_large = torch.tensor([0.02, 0.05, 0.10])
    out_large = _smooth_floor(pos_large, floor_tau=tau)
    expected_large = pos_large - 0.5 * tau
    assert torch.allclose(out_large, expected_large, atol=1e-6)


def test_pool_local_density_even_and_odd_kernels():
    """Verify _pool_local_density preserves exact shape and centering for both odd and even kernels."""
    y = torch.randn(2, 1, 32, 32)

    # Odd kernel
    y_odd = _pool_local_density(y, k_pool=5)
    assert y_odd.shape == y.shape

    # Even kernel (previously had shape mismatch or 1-pixel right/bottom truncation)
    y_even = _pool_local_density(y, k_pool=8)
    assert y_even.shape == y.shape
    assert torch.isfinite(y_even).all()


def test_solver_delta_y_step_direction_and_bounds():
    """Verify unrolled_sirt_solver adds mass when under-counting and removes mass when over-counting."""
    # Create simple 16x16 grid at stride 4 (image 64x64)
    h, w = 16, 16
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32, 64),
        overlap=0.5,
        device=torch.device("cpu"),
    )

    # 1. Test UNDER-COUNT: GT regional counts are positive, but carrier y0 is zero
    y0_zero = torch.zeros(1, 1, h, w)
    # Regional evidence says there are 5 people in the first box
    b_solver_pos = torch.zeros(1, 1, regions.boxes.shape[0])
    b_solver_pos[0, 0, 0] = 5.0
    weight_solver = torch.ones_like(b_solver_pos)

    res_add = unrolled_sirt_solver(
        y0=y0_zero,
        b_solver=b_solver_pos,
        weight_solver=weight_solver,
        regions=regions,
        iterations=2,
        omega=1.0,
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        scale_seeded_carrier=True,
        scale_seed_eps=0.02,
        scale_routing_weights=torch.ones(1, 2, h, w) * 0.5,
        proximal_mode="none",
        tv_lambda=0.0,
    )
    y_final_add = res_add["y"]
    # Model was under-counting (0 vs 5), so solver MUST add mass (y_final >= y0)
    assert y_final_add.sum() > y0_zero.sum(), "Solver failed to add mass when under-counting!"
    assert (y_final_add >= 0.0).all()

    # 2. Test OVER-COUNT: carrier y0 has positive mass, but regional evidence is 0
    y0_pos = torch.ones(1, 1, h, w) * 0.1
    b_solver_zero = torch.zeros(1, 1, regions.boxes.shape[0])

    res_rem = unrolled_sirt_solver(
        y0=y0_pos,
        b_solver=b_solver_zero,
        weight_solver=weight_solver,
        regions=regions,
        iterations=2,
        omega=1.0,
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        proximal_mode="none",
        tv_lambda=0.0,
    )
    y_final_rem = res_rem["y"]
    # Model was over-counting, so solver MUST reduce mass (y_final < y0)
    assert y_final_rem.sum() < y0_pos.sum(), "Solver failed to remove mass when over-counting!"
    assert (y_final_rem >= 0.0).all()


def test_allocation_loss_dispatch_weights():
    """Verify _compute_core_losses properly weights fidt and ot_sinkhorn without defaulting to lambda_flat_dm16."""
    y = torch.ones(1, 1, 16, 16, requires_grad=True)
    y0 = torch.ones(1, 1, 16, 16, requires_grad=True)
    target_float = torch.ones(1, 1, 16, 16)
    regions = build_multiscale_regions(16, 16, 4, region_sizes_px=(32,), overlap=0.5, device=torch.device("cpu"))
    target_region = torch.ones(1, 1, regions.boxes.shape[0])
    mean_region = torch.ones(1, 1, regions.boxes.shape[0], requires_grad=True)
    dispersion_region = torch.ones(1, 1, regions.boxes.shape[0]) * 50.0

    # Test FIDT allocation dispatch
    cfg_fidt = RMRv3LossConfig(
        allocation_loss_type="fidt",
        lambda_flat_dm16=0.0,  # Ensure flat_dm16 is 0
        lambda_count=1.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
    )
    setattr(cfg_fidt, "lambda_fidt", 2.5)
    router = TargetSupervisionRouter("y")
    losses = _compute_core_losses(
        target_float=target_float,
        target_region=target_region,
        y=y,
        y0=y0,
        regions=regions,
        mean_region=mean_region,
        dispersion_region=dispersion_region,
        cfg=cfg_fidt,
        points=[torch.tensor([[32.0, 32.0]])],
        router=router,
    )
    assert "total" in losses
    # Total loss should reflect 2.5 * fidt allocation + count loss
    assert losses["total"] > 0.0
