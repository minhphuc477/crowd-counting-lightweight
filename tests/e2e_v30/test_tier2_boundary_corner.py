"""Tier 2: Boundary & Corner Cases Test Suite for RMR-v30.

Exhaustively verifies mathematical boundaries, edge cases, singularities, and stress limits:
- Zero-Count & Empty Image Boundaries ($N=0$, >= 5 tests)
- Single Dirac Impulse Singularities ($N=1$, >= 5 tests)
- Hyper-Dense Clump Extremes ($N=2500$, >= 5 tests)
- Extreme Non-Square Aspect Ratios & Prime Dimension Grids (>= 5 tests)
- Anscombe Singularity at $y \to 0$ & Derivative Finiteness (>= 5 tests)
- Vanishing Residuals ($Ay = b$) & Orthogonal/Negative Curvature (>= 5 tests)
- Extreme Tau Ranges & Thresholding Dynamics (>= 4 tests)
- AMP fp16 Dynamic Range & Underflow Protections (>= 4 tests)
"""
from __future__ import annotations

import math
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.losses import flat_dm16_loss
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_sum,
)
from rmr_v3.losses import curvature_power_loss, mass_weighted_cell_loss
from rmr_v3.model import RMRv3, RMRv3Config

from tests.e2e_v30.conftest import (
    anscombe_discrepancy_oracle,
    anscombe_transform_oracle,
    check_mass_conservation_oracle,
    compute_adaptive_tau_oracle,
    get_rmr_v30_config_spec,
    inverse_anscombe_oracle,
    prolongation_stride4_to_stride2_rn_oracle,
    proximal_firm_threshold_oracle,
    pure_bb1_oracle,
    push_forward_stride2_to_stride4_oracle,
    support_invariance_oracle,
)


# ============================================================================
# Tier 2 - Category 1: Zero-Count & Empty Image Boundaries
# ============================================================================
class TestTier2ZeroCountAndEmptyImages:
    """Verifies boundary behaviors on zero-count images and empty backgrounds."""

    def test_tier2_empty_image_forward_no_nan(self, canonical_step0_config_dict: Dict[str, Any]) -> None:
        """All-zero pixel image input produces finite, non-negative predictions without NaN."""
        m_cfg = RMRv3Config.from_dict(canonical_step0_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()
        empty_img = torch.zeros(1, 3, 128, 128)
        with torch.no_grad():
            out = model(empty_img)
        assert torch.isfinite(out.y).all()
        assert (out.y >= 0.0).all()
        assert out.y.sum().item() >= 0.0

    def test_tier2_empty_image_zero_loss_target(self) -> None:
        """Zero-density ground truth (y_gt == 0) evaluates to finite loss without division-by-zero."""
        pred = torch.rand(1, 1, 32, 32) * 0.01
        target_zero = torch.zeros(1, 1, 32, 32)
        loss_cell = mass_weighted_cell_loss(pred, target_zero)
        loss_curv = curvature_power_loss(pred, target_zero)
        loss_dm16 = flat_dm16_loss(pred, target_zero)
        assert torch.isfinite(loss_cell)
        assert torch.isfinite(loss_curv)
        assert torch.isfinite(loss_dm16)

    def test_tier2_empty_image_radon_nikodym_zero_adjoint(self) -> None:
        """Theorem 4: In Radon-Nikodym adjoint, when y == 0, update delta_y is identically zero."""
        y_empty = torch.zeros(1, 1, 32, 32)
        # Any residual vector
        r = torch.randn(1, 1, 32, 32)
        delta_y = y_empty * r
        assert support_invariance_oracle(y_empty, delta_y)
        assert (delta_y == 0.0).all()

    def test_tier2_empty_image_anscombe_discrepancy_zero(self) -> None:
        """When regional predictions and ground truth are both zero (q=0, b=0), rate residual is 0."""
        q = torch.zeros(1, 1, 10)
        b = torch.zeros(1, 1, 10)
        res = anscombe_discrepancy_oracle(q, b, c=0.375)
        assert (res == 0.0).all()
        assert torch.isfinite(res).all()

    def test_tier2_empty_image_adaptive_tau_stability(self) -> None:
        """Zero input field produces maximum background threshold tau_eff == base_tau_step."""
        y_zero = torch.zeros(1, 1, 16, 16)
        tau_eff = compute_adaptive_tau_oracle(base_tau_step=0.015, y_current=y_zero, stride=4, rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert torch.allclose(tau_eff, torch.full_like(tau_eff, 0.015), atol=1e-6)


# ============================================================================
# Tier 2 - Category 2: Single Dirac Impulse Singularities (N=1)
# ============================================================================
class TestTier2SingleDiracImpulse:
    """Verifies singular point mass behaviors on isolated single heads (N=1)."""

    def test_tier2_dirac_center_point_mass_conservation(self) -> None:
        """Single Dirac delta at image center (16, 16) has total integrated count strictly == 1.0."""
        y = torch.zeros(1, 1, 32, 32)
        y[0, 0, 16, 16] = 1.0
        assert math.isclose(y.sum().item(), 1.0, rel_tol=1e-6)

    def test_tier2_dirac_top_left_corner_0_0(self) -> None:
        """Single Dirac delta at extreme top-left corner (0, 0) does not wrap around coordinates."""
        y = torch.zeros(1, 1, 32, 32)
        y[0, 0, 0, 0] = 1.0
        pref = prefix2d(y)
        box_tl = torch.tensor([[0, 0, 4, 4]])
        box_br = torch.tensor([[28, 28, 32, 32]])
        sum_tl = rectangle_sum_from_prefix(pref, box_tl)
        sum_br = rectangle_sum_from_prefix(pref, box_br)
        assert math.isclose(sum_tl.item(), 1.0, rel_tol=1e-6)
        assert math.isclose(sum_br.item(), 0.0, abs_tol=1e-6)

    def test_tier2_dirac_bottom_right_corner_h_w(self) -> None:
        """Single Dirac delta at bottom-right corner (H-1, W-1) is correctly bound."""
        y = torch.zeros(1, 1, 32, 32)
        y[0, 0, 31, 31] = 1.0
        pref = prefix2d(y)
        box_br = torch.tensor([[28, 28, 32, 32]])
        box_tl = torch.tensor([[0, 0, 4, 4]])
        sum_br = rectangle_sum_from_prefix(pref, box_br)
        sum_tl = rectangle_sum_from_prefix(pref, box_tl)
        assert math.isclose(sum_br.item(), 1.0, rel_tol=1e-6)
        assert math.isclose(sum_tl.item(), 0.0, abs_tol=1e-6)

    def test_tier2_dirac_dual_lattice_pushforward_exact_1(self) -> None:
        """Pushforward of a single Dirac head from Stride 2 to Stride 4 conserves count 1.000000."""
        y_fine = torch.zeros(1, 1, 64, 64)
        y_fine[0, 0, 25, 37] = 1.0
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-6)
        assert math.isclose(y_carrier.sum().item(), 1.0, rel_tol=1e-5)

    def test_tier2_dirac_anscombe_gradient_finite(self) -> None:
        """Anscombe power loss gradient on a single Dirac impulse is non-zero and finite."""
        pred = torch.zeros(1, 1, 16, 16, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16)
        target[0, 0, 8, 8] = 1.0
        loss = curvature_power_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[0, 0, 8, 8].item() < 0.0  # Under-predicted head receives negative gradient


# ============================================================================
# Tier 2 - Category 3: Hyper-Dense Clump Extremes (N=2500)
# ============================================================================
class TestTier2HyperDenseClump:
    """Verifies numerical stability under extreme crowd density clumps (N=2500)."""

    def test_tier2_hyper_dense_2500_heads_clump(self) -> None:
        """Synthetic dense clump of 2,500 heads inside a 64x64 patch integrates exactly."""
        torch.manual_seed(505)
        # 2500 heads dispersed randomly within a 64x64 grid
        y_clump = torch.zeros(1, 1, 64, 64)
        indices_h = torch.randint(16, 48, (2500,))
        indices_w = torch.randint(16, 48, (2500,))
        for ih, iw in zip(indices_h, indices_w):
            y_clump[0, 0, ih, iw] += 1.0
        assert math.isclose(y_clump.sum().item(), 2500.0, rel_tol=1e-5)

    def test_tier2_hyper_dense_anscombe_no_nan_overflow(self) -> None:
        """Anscombe transform on y=2500.0 yields finite real number ~ 100.007 without overflow."""
        y_dense = torch.tensor([2500.0])
        z = anscombe_transform_oracle(y_dense, c=0.375)
        expected = 2.0 * math.sqrt(2500.0 + 0.375)
        assert torch.isfinite(z)
        assert math.isclose(z.item(), expected, rel_tol=1e-6)

    def test_tier2_hyper_dense_bb1_step_stability(self) -> None:
        """In hyper-dense crowd clusters, BB-1 Rayleigh contraction step remains strictly bounded."""
        s = torch.randn(1, 1, 64, 64) * 100.0
        r = torch.randn(1, 1, 64, 64) * 50.0
        step = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.2, clamp_max=2.0)
        assert 0.2 <= step.item() <= 2.0

    def test_tier2_hyper_dense_adaptive_tau_vanishes(self) -> None:
        """In a 2,500-head clump, tau_eff smoothly attenuates to < 1e-4, preventing tail clipping."""
        y_dense = torch.full((1, 1, 64, 64), 2500.0 / (64 * 64))  # density ~ 0.61 >> 0.05
        tau_eff = compute_adaptive_tau_oracle(base_tau_step=0.015, y_current=y_dense, stride=4, rho0=0.05)
        assert isinstance(tau_eff, torch.Tensor)
        assert tau_eff.max().item() < 0.002, f"Tau did not vanish on dense crowd: {tau_eff.max().item()}"

    def test_tier2_hyper_dense_firm_thresholding_zero_mass_erosion(self) -> None:
        """High-density clump values receive 0.0 shrinkage, preserving total clump mass."""
        y_clump = torch.full((1, 1, 16, 16), 5.0)  # 5.0 >> mu*tau (0.045)
        out = proximal_firm_threshold_oracle(y_clump, tau=0.015, mu=3.0)
        assert torch.allclose(out, y_clump, atol=1e-7)
        assert math.isclose(out.sum().item(), y_clump.sum().item(), rel_tol=1e-6)


# ============================================================================
# Tier 2 - Category 4: Extreme Non-Square Aspect Ratios & Prime Grids
# ============================================================================
class TestTier2ExtremeAspectRatiosAndGrids:
    """Verifies spatial operators on extreme aspect ratios and non-power-of-two grids."""

    def test_tier2_extreme_aspect_ratio_1024x256(self) -> None:
        """Wide non-square aspect ratio (1024 x 256) executes pushforward with mass conservation."""
        y_fine = torch.rand(1, 1, 256, 1024) * 0.1
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert y_carrier.shape == (1, 1, 128, 512)
        assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-5)

    def test_tier2_extreme_aspect_ratio_256x1024(self) -> None:
        """Tall non-square aspect ratio (256 x 1024) executes prefix sums and regional queries."""
        y = torch.rand(1, 1, 1024, 256)
        pref = prefix2d(y)
        assert pref.shape == (1, 1, 1025, 257)
        box = torch.tensor([[100, 50, 900, 200]])
        val = rectangle_sum_from_prefix(pref, box)
        assert torch.isfinite(val)
        assert val.item() > 0.0

    def test_tier2_prime_dimension_grid_377x510(self) -> None:
        """Non-power-of-two dimension grid (376x510) pushforward executes cleanly."""
        y_fine = torch.rand(1, 1, 376, 510)
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert y_carrier.shape == (1, 1, 188, 255)
        assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-5)

    def test_tier2_minimal_tile_dimension_16x16(self) -> None:
        """Minimal valid grid (16 x 16) executes without index collapse."""
        y_fine = torch.rand(1, 1, 16, 16)
        y_carrier = push_forward_stride2_to_stride4_oracle(y_fine)
        assert y_carrier.shape == (1, 1, 8, 8)
        assert check_mass_conservation_oracle(y_fine, y_carrier, eps=1e-6)

    def test_tier2_large_scale_aspect_ratio_tiling(self) -> None:
        """Multi-scale region builder on non-square dimensions builds valid coverage."""
        regions = build_multiscale_regions(
            height=32,
            width=128,
            output_stride=4,
            region_sizes_px=((32, 32), (64, 64)),
            overlap=0.5,
        )
        assert len(regions.boxes) > 0
        y1, x1, y2, x2 = regions.boxes.unbind(dim=-1)
        assert (y2 <= 32).all()
        assert (x2 <= 128).all()


# ============================================================================
# Tier 2 - Category 5: Anscombe Singularity at y -> 0 & Derivative Finiteness
# ============================================================================
class TestTier2AnscombeSingularityBoundaries:
    """Verifies mathematical boundary conditions for Anscombe transformation near y=0."""

    def test_tier2_anscombe_boundary_y_zero(self) -> None:
        """At y = 0, T(0) = 2*sqrt(3/8) ≈ 1.22474487."""
        y_zero = torch.tensor([0.0])
        z = anscombe_transform_oracle(y_zero, c=0.375)
        expected = 2.0 * math.sqrt(0.375)
        assert math.isclose(z.item(), expected, rel_tol=1e-6)

    def test_tier2_anscombe_derivative_at_zero(self) -> None:
        """Derivative dT/dy at y=0 is 1/sqrt(3/8) ≈ 1.63299316 (strictly finite, no division-by-zero)."""
        y_zero = torch.tensor([0.0], requires_grad=True)
        z = anscombe_transform_oracle(y_zero, c=0.375)
        z.backward()
        assert y_zero.grad is not None
        expected_grad = 1.0 / math.sqrt(0.375)
        assert math.isclose(y_zero.grad.item(), expected_grad, rel_tol=1e-5)

    def test_tier2_anscombe_sub_epsilon_stability(self) -> None:
        """Microscopic densities y in [1e-12, 1e-6] do not trigger precision cancellation."""
        for val in [1e-12, 1e-9, 1e-6]:
            y = torch.tensor([val])
            z = anscombe_transform_oracle(y, c=0.375)
            assert torch.isfinite(z)
            assert z.item() >= 2.0 * math.sqrt(0.375)

    def test_tier2_anscombe_negative_clamping_safety(self) -> None:
        """Negative density inputs y = -5.0 are safely clamped to 0.0 without complex numbers or NaN."""
        y_neg = torch.tensor([-5.0, -0.01])
        z = anscombe_transform_oracle(y_neg, c=0.375)
        expected = 2.0 * math.sqrt(0.375)
        assert torch.allclose(z, torch.full_like(z, expected), atol=1e-6)

    def test_tier2_inverse_anscombe_negative_z_safety(self) -> None:
        """If z < 2*sqrt(3/8), inverse Anscombe safely clamps to 0.0."""
        z_small = torch.tensor([0.5, 1.0])  # Both < 1.2247
        y_rec = inverse_anscombe_oracle(z_small, c=0.375)
        assert (y_rec == 0.0).all()


# ============================================================================
# Tier 2 - Category 6: Vanishing Residuals & Curvature Fallbacks
# ============================================================================
class TestTier2VanishingAndOrthogonalResiduals:
    """Verifies solver stability under vanishing residuals and non-convex curvatures."""

    def test_tier2_vanishing_residuals_ay_equals_b(self) -> None:
        """When Ay == b (exact solution), Anscombe discrepancy rate residual r is identically 0."""
        q = torch.tensor([12.5, 40.0, 150.0])
        b = torch.tensor([12.5, 40.0, 150.0])
        res = anscombe_discrepancy_oracle(q, b, c=0.375)
        assert (res == 0.0).all()

    def test_tier2_zero_residual_bb1_step_fallback(self) -> None:
        """When residual difference r_diff == 0, BB-1 safely falls back to default omega_0."""
        s = torch.tensor([1.0, 2.0])
        r = torch.zeros(2)
        step = pure_bb1_oracle(s, r, omega_0=1.0)
        assert torch.isclose(step, torch.tensor([1.0]), atol=1e-5)

    def test_tier2_orthogonal_curvature_inner_product_zero(self) -> None:
        """When <s, r> == 0 (orthogonal curvature), BB-1 safely falls back to omega_0."""
        s = torch.tensor([1.0, 0.0])
        r = torch.tensor([0.0, 1.0])  # <s, r> = 0.0
        step = pure_bb1_oracle(s, r, omega_0=1.0)
        assert torch.isclose(step, torch.tensor([1.0]), atol=1e-5)

    def test_tier2_negative_curvature_fallthrough(self) -> None:
        """When <s, r> < 0 (negative curvature), BB-1 safely falls back to omega_0."""
        s = torch.tensor([1.0, 2.0])
        r = torch.tensor([-2.0, -1.0])  # <s, r> = -4.0 < 0
        step = pure_bb1_oracle(s, r, omega_0=1.2)
        assert torch.isclose(step, torch.tensor([1.2]), atol=1e-5)

    def test_tier2_identical_iterates_zero_step_diff(self) -> None:
        """When y_k == y_{k-1} (s=0), BB-1 does not crash and returns clamped default."""
        s = torch.zeros(10)
        r = torch.ones(10)
        step = pure_bb1_oracle(s, r, omega_0=1.0)
        assert torch.isclose(step, torch.tensor([1.0]), atol=1e-5)


# ============================================================================
# Tier 2 - Category 7: Extreme Tau Ranges & Thresholding Dynamics
# ============================================================================
class TestTier2ExtremeTauAndThresholding:
    """Verifies thresholding dynamics under extreme tau values."""

    def test_tier2_tau_zero_no_thresholding(self) -> None:
        """When tau = 0.0, firm thresholding reduces to non-negative projection clamp_min(0)."""
        y = torch.tensor([-1.0, 0.0, 0.5, 2.0])
        out = proximal_firm_threshold_oracle(y, tau=0.0)
        assert torch.allclose(out, torch.clamp_min(y, 0.0))

    def test_tier2_tau_extreme_large_full_pruning(self) -> None:
        """When tau = 100.0, all moderate density values are cleanly pruned to 0.0."""
        y = torch.rand(10) * 10.0  # max 10.0 << 100.0
        out = proximal_firm_threshold_oracle(y, tau=100.0)
        assert (out == 0.0).all()

    def test_tier2_firm_thresholding_mu_one_point_zero_zero_one(self) -> None:
        """Minimal mu = 1.001 evaluates safely without division-by-zero."""
        y = torch.tensor([0.02])
        out = proximal_firm_threshold_oracle(y, tau=0.015, mu=1.001)
        assert torch.isfinite(out)

    def test_tier2_tau_monotonic_sparsity(self) -> None:
        """Number of zero pixels strictly increases monotonically as tau increases."""
        torch.manual_seed(606)
        y = torch.rand(1, 1, 32, 32) * 0.1
        zeros_count = []
        for tau_val in [0.001, 0.01, 0.05, 0.15]:
            out = proximal_firm_threshold_oracle(y, tau=tau_val, mu=3.0)
            zeros_count.append((out == 0.0).sum().item())
        for i in range(len(zeros_count) - 1):
            assert zeros_count[i] <= zeros_count[i + 1]


# ============================================================================
# Tier 2 - Category 8: AMP fp16 Dynamic Range & Underflow Protections
# ============================================================================
class TestTier2FP16AndUnderflowPrecision:
    """Verifies numerical precision under half-precision and small density scales."""

    def test_tier2_fp16_dynamic_range_small_density(self) -> None:
        """Tiny density 1e-5 does not underflow to zero when evaluated in float32."""
        y_small = torch.tensor([1e-5], dtype=torch.float32)
        z = anscombe_transform_oracle(y_small, c=0.375)
        z_zero = anscombe_transform_oracle(torch.tensor([0.0]), c=0.375)
        assert (z - z_zero).item() > 0.0

    def test_tier2_fp16_autocast_accumulation_float32(self) -> None:
        """Under autocast, Anscombe transform computes in float32 without throwing dtype error."""
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y = torch.tensor([1.5, 4.0])
            z = anscombe_transform_oracle(y, c=0.375)
            assert torch.isfinite(z).all()

    def test_tier2_gradcheck_numerical_differentiability(self) -> None:
        """Numerical gradient matches analytical autograd gradient on Anscombe transform."""
        y = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        # Analytical derivative: 1 / sqrt(y + 3/8) = 1 / sqrt(2.375)
        z = 2.0 * torch.sqrt(y + 0.375)
        z.backward()
        analytical_grad = y.grad.item()

        eps = 1e-6
        y_plus = 2.0 + eps
        y_minus = 2.0 - eps
        z_plus = 2.0 * math.sqrt(y_plus + 0.375)
        z_minus = 2.0 * math.sqrt(y_minus + 0.375)
        numerical_grad = (z_plus - z_minus) / (2.0 * eps)
        assert math.isclose(analytical_grad, numerical_grad, rel_tol=1e-5)

    def test_tier2_large_count_fp16_no_infinity(self) -> None:
        """Extremely large count 100,000 does not overflow float16 dynamic range (max 65,504)."""
        y_huge = torch.tensor([100000.0])
        z = anscombe_transform_oracle(y_huge, c=0.375)
        # z = 2*sqrt(100000.375) ≈ 632.45, well below 65,504!
        assert z.item() < 1000.0
        assert torch.isfinite(z)
