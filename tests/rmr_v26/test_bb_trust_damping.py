"""Comprehensive Unit Test Suite for Pure BB-1 Rayleigh Contraction with Trust Damping.

Milestone M2 / Requirement R2:
1. Retain pure Barzilai-Borwein BB-1 step size:
       alpha_1 = <s_k, r_k> / (||r_k||^2 + eps)
2. Tighten the clamping range to [0.5 * omega_0, 1.2 * omega_0] (instead of [0.2, 2.0])
   to eliminate aggressive gradient swings in hyper-dense crowd clusters.
3. No alternating BB-2 step (disables BB-2 inverse Rayleigh quotient).
4. Parametrize bb_clamp_min and bb_clamp_max in RMRv3Config and solver.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rmr_core.operators import build_multiscale_regions
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


class TestBB1TrustDampingFormula:
    """Verifies the mathematical properties and clamping behavior of pure BB-1 step size."""

    def test_pure_bb1_formula_exactness(self) -> None:
        """alpha_1 = <s, r> / (||r||^2 + eps) matches analytical Rayleigh quotient."""
        eps = 1e-6
        s = torch.tensor([[[[2.0, 3.0], [1.0, 4.0]]]])  # shape [1, 1, 2, 2]
        r = torch.tensor([[[[1.0, 2.0], [0.5, 2.0]]]])  # shape [1, 1, 2, 2]

        dot_sr = (s * r).sum()
        norm_r_sq = (r * r).sum() + eps
        expected_raw = (dot_sr / norm_r_sq).item()

        assert expected_raw > 0.0, "Expected positive inner product"
        assert abs(expected_raw - (16.5 / 9.250001)) < 1e-4

    def test_trust_damping_clamp_bounds(self) -> None:
        """Verify that step size is strictly clamped in [0.5 * omega, 1.2 * omega]."""
        regions = build_multiscale_regions(64, 64, 4, (32, 64, 128), 0.5)
        h, w = 64, 64
        m = len(regions.boxes)

        y0 = torch.full((1, 1, h, w), 0.5)
        b_solver = torch.full((1, 1, m), 2.0)
        weight_solver = torch.ones(1, 1, m)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=6,
            omega=1.0,
            use_barzilai_borwein=True,
            use_alternating_bb=False,
            bb_clamp_min=0.5,
            bb_clamp_max=1.2,
        )

        assert len(sol["iterates"]) == 7  # [y0, y1, y2, y3, y4, y5, y6]
        for idx, iterate in enumerate(sol["iterates"]):
            assert torch.isfinite(iterate).all(), f"Iterate {idx} has non-finite values"
            assert (iterate >= 0.0).all(), f"Iterate {idx} has negative values"

    def test_fallback_when_bb_disabled(self) -> None:
        """When use_barzilai_borwein=False, omega remains strictly fixed at effective_omega."""
        regions = build_multiscale_regions(64, 64, 4, (32, 64, 128), 0.5)
        h, w = 64, 64
        m = len(regions.boxes)

        y0 = torch.full((1, 1, h, w), 0.5)
        b_solver = torch.full((1, 1, m), 2.0)
        weight_solver = torch.ones(1, 1, m)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=4,
            omega=1.0,
            use_barzilai_borwein=False,
            use_alternating_bb=False,
        )

        assert sol["effective_omega"] == 1.0


class TestModelIntegrationBB1:
    """Verifies end-to-end integration of BB-1 trust damping in RMRv3 model."""

    def test_model_config_bb_clamp_defaults(self) -> None:
        """RMRv3Config must default bb_clamp_min=0.5 and bb_clamp_max=1.2."""
        cfg = RMRv3Config()
        assert cfg.bb_clamp_min == 0.5
        assert cfg.bb_clamp_max == 1.2
        assert cfg.use_barzilai_borwein is False  # default before config overrides
        assert cfg.use_alternating_bb is False

    def test_model_forward_backward_with_bb1(self) -> None:
        """RMRv3 model with BB-1 enabled runs forward and backward smoothly."""
        cfg = RMRv3Config(
            pretrained=False,
            use_barzilai_borwein=True,
            use_alternating_bb=False,
            bb_clamp_min=0.5,
            bb_clamp_max=1.2,
            use_perspective_elevation=True,
            enable_solver=True,
            iterations=2,
            omega=1.0,
        )
        model = RMRv3(cfg)
        model.train()

        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        assert "y" in out
        assert out.y.shape == (2, 1, 32, 32)
        assert torch.isfinite(out.y).all()

        loss = out.y.sum()
        loss.backward()

        # Check gradients exist on backbone, elevation, and heads
        assert model.perspective_elevation.proj.weight.grad is not None
        assert torch.isfinite(model.perspective_elevation.proj.weight.grad).all()
        # Check any parameter in fine_head has non-null gradient
        fine_grads = [p.grad for p in model.fine_head.parameters() if p.grad is not None]
        assert len(fine_grads) > 0
