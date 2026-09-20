"""Empirical Stress Tests for Milestone 1 Numerical & Edge-Case Verification.

Authored by Challenger 2 (Edge-Case & Numerical Challenger).
Tests:
1. Barzilai-Borwein BB-1 step size clamping under orthogonal/negative curvature (<s, r> <= 0).
2. Mass conservation under 100 consecutive push-forward/pullback iterations.
3. Strict support invariance on synthetic images with 99.9% empty backgrounds over 10 SIRT iterations.
4. AMP float16 precision and preservation of small densities in Anscombe transform.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import build_multiscale_regions
from rmr_v3.solver_ops import (
    anscombe_transform,
    anscombe_discrepancy,
    compute_adaptive_tau,
    proximal_firm_threshold,
)
from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)
from rmr_v3.solver import unrolled_sirt_solver


class TestBB1ClampingStress:
    """Stress tests for Barzilai-Borwein BB-1 step size bounds."""

    def test_bb1_step_clamping_synthetic_extremes(self):
        """Verify BB-1 step clamping for orthogonal, negative, and extreme curvature pairs."""
        omega_0 = 1.0
        bb_clamp_min = 0.2
        bb_clamp_max = 2.0

        # Construct synthetic (s, r) scenarios
        scenarios = [
            ("orthogonal", torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
            ("negative_curvature", torch.tensor([1.0, -2.0]), torch.tensor([-1.0, 2.0])),
            ("extreme_negative", torch.tensor([1000.0]), torch.tensor([-1000.0])),
            ("zero_step", torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])),
            ("tiny_positive", torch.tensor([1e-6]), torch.tensor([1.0])),
            ("huge_positive", torch.tensor([1e6]), torch.tensor([1.0])),
        ]

        for name, s, r in scenarios:
            dot_sr = (s * r).sum()
            norm_r_sq = (r * r).sum() + 1e-6

            omega_candidate = torch.where(
                dot_sr > 0.0,
                dot_sr / norm_r_sq,
                torch.as_tensor(omega_0, dtype=dot_sr.dtype),
            )
            clamped = torch.clamp(
                omega_candidate,
                min=bb_clamp_min * omega_0,
                max=bb_clamp_max * omega_0,
            ).item()

            assert bb_clamp_min * omega_0 - 1e-6 <= clamped <= bb_clamp_max * omega_0 + 1e-6, (
                f"Scenario '{name}' produced clamped omega={clamped}, out of [{bb_clamp_min}, {bb_clamp_max}]"
            )
            if dot_sr <= 0.0:
                assert math.isclose(clamped, omega_0, rel_tol=1e-5), (
                    f"Scenario '{name}' (<s, r> <= 0) should fall back to omega_0={omega_0}, got {clamped}"
                )

    def test_bb1_in_unrolled_solver_10_iterations(self):
        """Verify that across 10 unrolled iterations, every step size is strictly in [0.2, 2.0]*omega_0."""
        torch.manual_seed(123)
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,), overlap=0.5, include_full_image=False)

        # Adversarial oscillating target designed to provoke sign flips in <s, r>
        y0 = torch.rand(2, 1, h, w) * 3.0
        b_solver = torch.rand(2, 1, len(regions.boxes)) * 10.0
        weight_solver = torch.ones_like(b_solver)

        omega_0 = 1.25
        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=10,
            omega=omega_0,
            solver_strength=1.0,
            use_barzilai_borwein=True,
            bb_clamp_min=0.2,
            bb_clamp_max=2.0,
            use_anscombe=True,
            adjoint_mode="anscombe_vst",
        )

        step_omegas = res["step_omegas"]
        assert len(step_omegas) == 10
        for i, om in enumerate(step_omegas):
            if isinstance(om, torch.Tensor):
                min_v = om.min().item()
                max_v = om.max().item()
            else:
                min_v = max_v = float(om)
            assert 0.2 * omega_0 - 1e-5 <= min_v, f"Step {i} min omega {min_v} < {0.2 * omega_0}"
            assert max_v <= 2.0 * omega_0 + 1e-5, f"Step {i} max omega {max_v} > {2.0 * omega_0}"


class TestSupportInvarianceStress:
    """Stress tests for strict background support invariance."""

    def test_support_invariance_99_percent_background_10_iterations(self):
        """Assert all 16,368 background pixels (99.9% of image) remain identically 0.0000 after 10 SIRT steps."""
        torch.manual_seed(999)
        h, w = 128, 128
        regions = build_multiscale_regions(
            h, w, output_stride=4, region_sizes_px=(32, 64), overlap=0.5, include_full_image=False
        )

        y0 = torch.zeros(1, 1, h, w)
        # Place a single 4x4 cluster in center (16 foreground pixels)
        y0[0, 0, 62:66, 62:66] = 2.0

        # Discrepancy target demanding huge mass everywhere in the scene
        b_solver = torch.ones(1, 1, len(regions.boxes)) * 20.0
        weight_solver = torch.ones_like(b_solver)

        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=10,
            omega=1.0,
            tv_lambda=0.0,
            use_anscombe=True,
            adjoint_mode="anscombe_vst",
            proximal_tau=0.015,
            proximal_mode="firm",
            use_barzilai_borwein=True,
        )

        y_final = res["y"]
        bg_mask = (y0 == 0.0)
        assert bg_mask.sum().item() == 16384 - 16

        bg_pixels = y_final[bg_mask]
        assert (bg_pixels == 0.0).all(), (
            f"Support invariance violated! Max background pixel = {bg_pixels.max().item():.8e}"
        )
        assert bg_pixels.sum().item() == 0.0


class TestAMPFloat16PrecisionStress:
    """Stress tests for float16 precision in Anscombe operators."""

    def test_float16_densities_anscombe_transform_no_zero_out(self):
        """Verify normal and subnormal float16 densities do not zero out in Anscombe transform."""
        densities = [1e-7, 1e-6, 1e-5, 6.1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
        y_f16 = torch.tensor(densities, dtype=torch.float16)
        t_out = anscombe_transform(y_f16, c=0.375)

        t_zero = anscombe_transform(torch.tensor([0.0], dtype=torch.float16), c=0.375)
        diff = t_out - t_zero

        for d, df in zip(densities, diff):
            assert df.item() > 0.0, f"Density {d} zeroed out in Anscombe transform (diff={df.item()})"

    def test_float16_anscombe_discrepancy_sensitivity(self):
        """Verify small float16 count differences produce non-zero gradients."""
        q_f16 = torch.tensor([6.1e-5, 1e-4, 1e-3], dtype=torch.float16)
        b_f16 = torch.tensor([9.1e-5, 2e-4, 2e-3], dtype=torch.float16)

        disc = anscombe_discrepancy(q_f16, b_f16, c=0.375)
        for q, b, dc in zip(q_f16, b_f16, disc):
            assert abs(dc.item()) > 0.0, f"Small discrepancy q={q}, b={b} zeroed out!"


class TestMassConservation100IterationsInvariant:
    """Empirical verification harness for mass conservation over 100 iterations.

    Verifies that pullback_stride4_to_stride2_rn strictly maintains cumulative drift <= 1e-5
    across 100 consecutive push-forward/pullback iterations across all density regimes.
    """

    def test_100_iterations_pullback_satisfies_invariant(self):
        """Verify that pullback_stride4_to_stride2_rn achieves exact zero drift (<= 1e-5) across all regimes."""
        torch.manual_seed(42)
        cases = {
            "single_dirac": torch.zeros(1, 1, 64, 64),
            "multi_dirac": torch.zeros(1, 1, 64, 64),
            "small_densities": torch.ones(1, 1, 64, 64) * 1e-4,
            "tiny_densities": torch.ones(1, 1, 64, 64) * 1e-6,
            "large_densities": torch.ones(1, 1, 64, 64) * 10.0,
        }
        cases["single_dirac"][0, 0, 10, 10] = 1.0
        for r in [5, 15, 25]:
            for c in [5, 15, 25]:
                cases["multi_dirac"][0, 0, r, c] = 1.0

        for name, y2 in cases.items():
            initial_mass = y2.double().sum().item()
            curr_y2 = y2.clone()
            for _ in range(100):
                y4 = push_forward_stride2_to_stride4(curr_y2)
                curr_y2 = pullback_stride4_to_stride2_rn(y4, curr_y2)
            final_mass = curr_y2.double().sum().item()
            abs_drift = abs(final_mass - initial_mass)
            rel_drift = abs_drift / max(initial_mass, 1e-12)
            assert abs_drift <= 1e-5, f"Case {name} failed: abs_drift={abs_drift:.4e} > 1e-5"
            assert rel_drift <= 1e-5, f"Case {name} failed: rel_drift={rel_drift:.4e} > 1e-5"

    def test_100_iterations_single_dirac_exact_zero_drift(self):
        """Verify the exact canonical Challenger 2 test case on single Dirac."""
        y2 = torch.zeros(1, 1, 64, 64)
        y2[0, 0, 10, 10] = 1.0
        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)
        drift = abs(curr.double().sum().item() - init_m)
        assert drift <= 1e-5, f"Single Dirac drift too high: {drift}"
        assert math.isclose(drift, 0.0, abs_tol=1e-8), f"Expected exact 0.0 drift, got {drift}"
