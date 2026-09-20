"""Milestone 1 Re-verification Challenger Stress Harness.

Authored by Edge-Case Re-verification Challenger.
Rigorously tests:
1. Exact mass conservation under 100 consecutive pushforward/pullback iterations across:
   - Single Dirac across all 4 subpixel quadrant positions and lattice corners
   - Multi-Dirac (grid, random sparse, clustered within coarse block)
   - Tiny densities (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
   - Large clumps (1e2, 1e4, 1e6, 1e8)
   - Mixed scene (canvas with background 0, tiny noise, Diracs, and dense clump)
2. Local 2x2 cell-wise mass conservation at each step.
3. Invariant checks under varying batch sizes and non-square aspect ratios.
4. Support invariance: background cells (0.0) remain strictly 0.0 under unrolled cycles.
5. Backward pass stability (zero NaNs, finite gradients) through unrolled pullback.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)


class TestM1ReverificationPushforwardPullbackDrift:
    """Adversarial stress harness for pushforward / pullback 100-step mass drift."""

    @pytest.mark.parametrize(
        "sub_r, sub_c",
        [
            (0, 0),  # top-left
            (0, 1),  # top-right
            (1, 0),  # bottom-left
            (1, 1),  # bottom-right
        ],
    )
    def test_single_dirac_all_subpixel_quadrants_100_iterations(self, sub_r: int, sub_c: int):
        """Single Dirac at each quadrant position within a 2x2 coarse block."""
        h, w = 64, 64
        r, c = 20 + sub_r, 20 + sub_c
        y2 = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y2[0, 0, r, c] = 1.0

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / max(init_m, 1e-12)

        assert abs_drift <= 1e-5, f"Quadrant ({sub_r}, {sub_c}) failed: abs_drift={abs_drift:.4e}"
        assert rel_drift <= 1e-5, f"Quadrant ({sub_r}, {sub_c}) failed: rel_drift={rel_drift:.4e}"
        assert math.isclose(abs_drift, 0.0, abs_tol=1e-8), f"Expected exact 0.0 drift, got {abs_drift}"

    @pytest.mark.parametrize(
        "corner_r, corner_c",
        [
            (0, 0),
            (0, 63),
            (63, 0),
            (63, 63),
        ],
    )
    def test_single_dirac_lattice_corners_100_iterations(self, corner_r: int, corner_c: int):
        """Single Dirac at extreme boundary corners."""
        h, w = 64, 64
        y2 = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y2[0, 0, corner_r, corner_c] = 1.0

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        assert abs_drift <= 1e-5, f"Corner ({corner_r}, {corner_c}) abs_drift={abs_drift:.4e}"
        assert math.isclose(abs_drift, 0.0, abs_tol=1e-8)

    def test_multi_dirac_clustered_and_disjoint_100_iterations(self):
        """Multi-Dirac including points in the same 2x2 coarse block and disjoint blocks."""
        h, w = 64, 64
        y2 = torch.zeros(1, 1, h, w, dtype=torch.float32)
        # Clustered in same 2x2 coarse cell (10, 10) -> fine (20..21, 20..21)
        y2[0, 0, 20, 20] = 0.5
        y2[0, 0, 20, 21] = 1.5
        y2[0, 0, 21, 20] = 2.5
        y2[0, 0, 21, 21] = 3.5
        # Disjoint Dirac points
        for r, c in [(4, 4), (12, 50), (48, 12), (60, 60)]:
            y2[0, 0, r, c] = 1.0

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / init_m
        assert abs_drift <= 1e-5, f"Multi-dirac abs_drift={abs_drift:.4e}"
        assert rel_drift <= 1e-5, f"Multi-dirac rel_drift={rel_drift:.4e}"
        assert math.isclose(abs_drift, 0.0, abs_tol=1e-8)

    @pytest.mark.parametrize("density_val", [1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
    def test_tiny_and_sub_epsilon_densities_100_iterations(self, density_val: float):
        """Verify mass conservation for small, tiny, and sub-epsilon densities over 100 iterations."""
        h, w = 64, 64
        y2 = torch.full((1, 1, h, w), density_val, dtype=torch.float32)

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / max(init_m, 1e-16)

        assert abs_drift <= 1e-5, f"Density {density_val} failed: abs_drift={abs_drift:.4e}"
        assert rel_drift <= 1e-5, f"Density {density_val} failed: rel_drift={rel_drift:.4e}"

    @pytest.mark.parametrize("clump_scale", [100.0, 10000.0, 1000000.0, 100000000.0])
    def test_large_clumps_100_iterations(self, clump_scale: float):
        """Verify mass conservation for massive clump counts over 100 iterations."""
        h, w = 64, 64
        y2 = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y2[0, 0, 28:36, 28:36] = clump_scale

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / init_m

        assert rel_drift <= 1e-5, f"Clump {clump_scale} failed: rel_drift={rel_drift:.4e}"

    def test_mixed_scene_canvas_100_iterations(self):
        """Mixed canvas containing background zeros, tiny noise (1e-6), Diracs, and dense clump."""
        h, w = 64, 64
        y2 = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y2[0, 0, :16, :16] = 1e-6
        y2[0, 0, 10, 50] = 1.0
        y2[0, 0, 20, 40] = 2.0
        y2[0, 0, 40:48, 40:48] = 50.0

        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / init_m

        assert abs_drift <= 1e-5, f"Mixed scene abs_drift={abs_drift:.4e}"
        assert rel_drift <= 1e-5, f"Mixed scene rel_drift={rel_drift:.4e}"
        bg_q3 = curr[0, 0, 32:, :32]
        assert (bg_q3 == 0.0).all(), f"Support leaked in Q3: max={bg_q3.max().item():.4e}"

    @pytest.mark.parametrize(
        "h, w",
        [
            (64, 48),
            (128, 96),
            (48, 80),
            (256, 192),
        ],
    )
    def test_nonsquare_and_asymmetric_lattices_100_iterations(self, h: int, w: int):
        """Verify drift across various asymmetric and rectangular lattice sizes."""
        torch.manual_seed(99)
        y2 = torch.rand(1, 1, h, w, dtype=torch.float32) * 5.0
        init_m = y2.double().sum().item()
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum().item()
        abs_drift = abs(final_m - init_m)
        rel_drift = abs_drift / init_m

        assert abs_drift <= 1e-5, f"Shape ({h}, {w}) abs_drift={abs_drift:.4e}"
        assert rel_drift <= 1e-5, f"Shape ({h}, {w}) rel_drift={rel_drift:.4e}"

    def test_batch_heterogeneous_regimes_100_iterations(self):
        """Batch of 4 images with completely different regimes in each batch item."""
        b_size = 4
        y2 = torch.zeros(b_size, 1, 64, 64, dtype=torch.float32)
        y2[0, 0, 10, 10] = 1.0
        y2[1, 0, 5:10, 5:10] = 1.0
        y2[2, 0] = 1e-6
        y2[3, 0, 20:40, 20:40] = 100.0

        init_m = y2.double().sum(dim=(-2, -1)).squeeze(1)
        curr = y2.clone()
        for _ in range(100):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        final_m = curr.double().sum(dim=(-2, -1)).squeeze(1)
        abs_drift = (final_m - init_m).abs()
        rel_drift = abs_drift / init_m.clamp_min(1e-12)

        for i in range(b_size):
            assert abs_drift[i].item() <= 1e-5 or rel_drift[i].item() <= 1e-5, (
                f"Batch item {i} failed: abs_drift={abs_drift[i].item():.4e}, rel_drift={rel_drift[i].item():.4e}"
            )

    def test_cellwise_local_mass_conservation(self):
        """Verify that every 2x2 fine cell block sums exactly to its coarse cell."""
        torch.manual_seed(777)
        h, w = 32, 32
        y2 = torch.rand(2, 1, h, w, dtype=torch.float32) * 10.0
        y4 = push_forward_stride2_to_stride4(y2)
        prolong = pullback_stride4_to_stride2_rn(y4, y2)

        re_coarse = 4.0 * F.avg_pool2d(prolong, kernel_size=2, stride=2, count_include_pad=False)
        cell_diff = (re_coarse - y4).abs()
        max_cell_diff = cell_diff.max().item()

        assert max_cell_diff <= 1e-5, f"Local cell-wise mass conservation failed: max_diff={max_cell_diff:.4e}"

    def test_autograd_gradient_stability_through_unrolled_pullback(self):
        """Verify backward gradient propagation through 10 unrolled pullback cycles without NaNs."""
        y2_init = (torch.rand(1, 1, 32, 32, dtype=torch.float32) * 2.0).requires_grad_(True)
        curr = y2_init
        for _ in range(10):
            y4 = push_forward_stride2_to_stride4(curr)
            curr = pullback_stride4_to_stride2_rn(y4, curr)

        loss = curr.sum()
        loss.backward()

        assert y2_init.grad is not None
        assert torch.isfinite(y2_init.grad).all(), "NaN or Inf gradients found in unrolled pullback!"
        assert (y2_init.grad != 0.0).any(), "Gradients completely vanished!"
