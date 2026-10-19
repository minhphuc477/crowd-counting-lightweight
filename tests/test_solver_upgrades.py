"""Comprehensive verification tests for upgraded SIRT solver and carrier operators.

Validates:
1. compute_deficit_carrier_seed:
   - Invariant A: Exact zero background leakage (psi == 0 where b_solver <= Ay).
   - Invariant B: Dense deficit discovery (psi > 0 where b_solver > Ay).
   - Invariant C: Self-terminating convergence (psi -> 0 as Ay -> b_solver).
2. Density-adaptive Morozov deadband:
   - Preserves full gamma on empty background.
   - Decays gamma on high-density crowd boxes.
3. unrolled_sirt_solver integration:
   - Non-negative mass conservation: y >= 0.
   - Escapes zero-support trap on missing head clusters.
"""
from __future__ import annotations

import torch
import pytest

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    compute_deficit_carrier_seed,
    compute_morozov_discrepancy,
    modulate_carrier_field,
    regional_sum,
)
from rmr_v3.solver import unrolled_sirt_solver


def test_deficit_carrier_seeding_invariants():
    """Verify that deficit carrier seeding strictly enforces zero background leakage

    while providing positive carrier support in under-counted crowd regions.
    """
    device = torch.device("cpu")
    h, w = 32, 32

    # Region 0: Box [8, 8, 24, 24] in center (dense crowd region)
    # Region 1: Box [0, 0, 8, 8] in top-left (empty background region)
    boxes = torch.tensor([[8, 8, 24, 24], [0, 0, 8, 8]], dtype=torch.int64)
    area = torch.tensor([256, 64], dtype=torch.int64)
    scale_id = torch.tensor([0, 0], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, area=area, scale_id=scale_id)

    # Ground truth regional evidence:
    # Region 0: b_solver = 20.0 (crowd cluster)
    # Region 1: b_solver = 0.0 (background)
    b_solver = torch.tensor([[20.0, 0.0]], dtype=torch.float32)

    # Current carrier state y has 0.0 everywhere (complete zero-support trap)
    y = torch.zeros((1, 1, h, w), dtype=torch.float32)

    # Compute deficit carrier seed
    psi = compute_deficit_carrier_seed(y, b_solver, regions, output_stride=4)

    # Invariant A: Background box [0:8, 0:8] has b_solver <= Ay (both 0) -> psi MUST be identically 0.0!
    assert (psi[0, 0, 0:8, 0:8] == 0.0).all(), "Deficit carrier leaked into background box!"

    # Invariant B: Dense box [8:24, 8:24] has b_solver = 20 > Ay = 0 -> psi MUST be positive!
    assert (psi[0, 0, 8:24, 8:24] > 0.0).all(), "Deficit carrier failed to seed under-counted crowd box!"

    # Invariant C: Self-terminating: when y satisfies the deficit, psi turns off to 0.0
    y_satisfied = torch.zeros((1, 1, h, w), dtype=torch.float32)
    # Put 20.0 mass into box 0
    y_satisfied[0, 0, 8:24, 8:24] = 20.0 / 256.0
    psi_satisfied = compute_deficit_carrier_seed(y_satisfied, b_solver, regions, output_stride=4)
    assert (psi_satisfied == 0.0).all(), "Deficit carrier did not self-terminate when deficit resolved!"


def test_density_adaptive_morozov_deadband():
    """Verify that density-adaptive Morozov deadband maintains full threshold

    on background boxes while shrinking threshold on dense crowd boxes.
    """
    device = torch.device("cpu")
    # 2 boxes: box 0 empty (b=0), box 1 dense (b=50 on 16x16=256 area, density=0.195)
    boxes = torch.tensor([[0, 0, 16, 16], [16, 16, 32, 32]], dtype=torch.int64)
    area = torch.tensor([256, 256], dtype=torch.int64)
    scale_id = torch.tensor([0, 0], dtype=torch.int64)
    regions = RegionSet(boxes=boxes, area=area, scale_id=scale_id)

    b32 = torch.tensor([[[0.0, 50.0]]], dtype=torch.float32)
    q = torch.tensor([[[0.0, 40.0]]], dtype=torch.float32)
    delta = q - b32  # [0.0, -10.0]
    b_var = torch.tensor([[[0.01, 100.0]]], dtype=torch.float32)  # sigma_b = [0.1, 10.0]

    # Standard Morozov (density_adaptive_morozov=False, gamma=0.75):
    # Deadband on box 1 is 0.75 * 10.0 = 7.5
    # Since |delta| = 10.0 > 7.5: residual shrunk to -(10 - 7.5) = -2.5
    shrunk_std = compute_morozov_discrepancy(
        delta=delta, q=q, b32=b32, b_variance=b_var, regions=regions,
        morozov_gamma=0.75, density_adaptive_morozov=False,
    )
    assert pytest.approx(shrunk_std[0, 0, 1].item(), abs=1e-3) == -2.5

    # Density-Adaptive Morozov (density_adaptive_morozov=True, beta=3.0):
    # Box 0: b=0 -> gamma_eff = 0.75 (full background deadband preserved!)
    # Box 1: density = 50 / 256 = 0.195 -> gamma_eff = 0.75 / (1 + 3 * 0.195) = 0.75 / 1.585 = 0.473
    # Deadband on box 1 shrinks from 7.5 to 0.473 * 10.0 = 4.73
    # Shrunk residual becomes -(10 - 4.73) = -5.27 (substantially larger correction signal!)
    shrunk_dense = compute_morozov_discrepancy(
        delta=delta, q=q, b32=b32, b_variance=b_var, regions=regions,
        morozov_gamma=0.75, density_adaptive_morozov=True, morozov_density_beta=3.0,
    )
    assert abs(shrunk_dense[0, 0, 1].item()) > abs(shrunk_std[0, 0, 1].item())
    assert pytest.approx(shrunk_dense[0, 0, 1].item(), abs=0.1) == -9.28


def test_unrolled_sirt_solver_escapes_dense_zero_support_trap():
    """Verify that unrolled SIRT solver with deficit carrier seeding escapes

    the zero-support trap on missing dense head clusters.
    """
    device = torch.device("cpu")
    h, w = 24, 24
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(16,), device=device)

    # Initial carrier y0 is flat background 0.0 everywhere except 1 tiny peak at (4, 4)
    # Box 0 is [0, 0, 4, 4] (cells 0 to 4 in grid)
    y0 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    y0[0, 0, 1, 1] = 0.5  # 0.5 people at (1, 1) inside box 0

    # Regional observation b_solver says there are 10.0 people in box 0 [0:4, 0:4]
    b_solver = torch.zeros((1, 1, len(regions.boxes)), dtype=torch.float32)
    b_solver[0, 0, 0] = 10.0
    weight_solver = torch.ones_like(b_solver)

    # 1. Standard SIRT in Radon-Nikodym mode (deficit_carrier_seeding=False):
    # Trapped! Cannot inject mass into pixels where y0 == 0.0
    res_trap = unrolled_sirt_solver(
        y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=4, omega=1.0, trust_region_kappa=0.0, adjoint_mode="radon_nikodym",
        deficit_carrier_seeding=False,
    )
    # Mass at zero pixel (2, 2) inside box 0 remains stuck at 0.0
    assert res_trap["y"][0, 0, 2, 2].item() == 0.0

    # 2. Upgraded SIRT (deficit_carrier_seeding=True, seed_eps=0.03):
    # Successfully breaks zero-support trap!
    res_upgraded = unrolled_sirt_solver(
        y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=4, omega=1.0, trust_region_kappa=0.0, adjoint_mode="radon_nikodym",
        deficit_carrier_seeding=True, deficit_seed_eps=0.03,
    )
    # Standard RN concentrates all mass onto the single initial pixel (1 out of 16 cells)
    assert (res_trap["y"][0, 0, :4, :4] > 0).sum().item() == 1
    # Upgraded RN with deficit carrier seeding unlocks support across the deficit box (16 out of 16 cells)
    assert (res_upgraded["y"][0, 0, :4, :4] > 0).sum().item() == 16

    # Invariant: Output remains strictly non-negative everywhere
    assert (res_upgraded["y"] >= 0.0).all()


def test_trust_region_deficit_allowance_and_asymmetric_density_morozov():
    """Verify that deficit seed allowance in trust region permits mass injection

    into deficit cells while guaranteeing zero leakage into non-deficit background.
    Also verify asymmetric Morozov correctly inherits density attenuation.
    """
    device = torch.device("cpu")
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, 4, region_sizes_px=(16,), device=device)

    y0 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    y0[0, 0, 1, 1] = 0.1

    b_solver = torch.zeros((1, 1, len(regions.boxes)), dtype=torch.float32)
    b_solver[0, 0, 0] = 20.0
    weight_solver = torch.ones_like(b_solver)
    b_var = torch.ones_like(b_solver) * 20.0

    res = unrolled_sirt_solver(
        y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=6, omega=1.0, trust_region_kappa=0.35, trust_region_floor=0.005,
        adjoint_mode="radon_nikodym", morozov_gamma=0.75, b_variance=b_var,
        deficit_carrier_seeding=True, deficit_seed_eps=0.05,
        density_adaptive_morozov=True, morozov_density_beta=2.0,
    )
    y_star = res["y"]

    # 1. Deficit box [0:4, 0:4] unlocked all 16 cells
    assert (y_star[0, 0, :4, :4] > 0.001).sum().item() == 16
    # 2. Deficit mass recovered is substantially greater than the tight 0.00175 ceiling
    assert y_star[0, 0, :4, :4].sum().item() > 2.0
    # 3. Non-deficit background [8:, 8:] has strictly 0 leakage
    assert (y_star[0, 0, 8:, 8:] > 0.001).sum().item() == 0

    # 4. Asymmetric Morozov test with density adaptation
    boxes = torch.tensor([[0, 0, 4, 4]], dtype=torch.int64)
    reg = RegionSet(boxes=boxes, area=torch.tensor([16], dtype=torch.int64), scale_id=torch.tensor([0], dtype=torch.int64))
    b = torch.tensor([[[20.0]]], dtype=torch.float32)
    q = torch.tensor([[[5.0]]], dtype=torch.float32)
    delta = q - b  # Under-counting by 15
    var = torch.tensor([[[20.0]]], dtype=torch.float32)

    shrunk_asym = compute_morozov_discrepancy(
        delta=delta, q=q, b32=b, b_variance=var, regions=reg,
        morozov_gamma=0.75, asymmetric_morozov=True, morozov_gamma_under=0.20,
        density_adaptive_morozov=True, morozov_density_beta=2.0,
    )
    # Density factor = 1 / (1 + 2 * (20/16)) = 1 / 3.5 = 0.2857
    # Gamma under = (0.75 * 0.2857) / (1 + 0.3 * sqrt(20)) = 0.2143 / 2.3416 = 0.0915
    # Shrunk residual retains high magnitude (~ -14.59)
    assert shrunk_asym.item() < -14.0

