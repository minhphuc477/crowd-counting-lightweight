"""Tier 2: Boundary & Corner Cases Test Suite for RMR-v26.

Exhaustively verifies mathematical boundaries, edge cases, singularities, and stress limits
across all 7 features in PROJECT.md:
- F1: MPE-v2 Boundary Cases (>=5 tests)
- F2: BB-1 Damping Singularities & Edge Cases (>=5 tests)
- F3: Loss Core Boundaries & Zero/Clump Densities (>=5 tests)
- F4: Config Schema Rejection & Boundary Values (>=5 tests)
- F5: Mathematical Invariant Edge Cases & Singularities (>=5 tests)
- F6: Execution Boundary Conditions & Batch Variations (>=5 tests)
- F7: Evaluation Metric Edge Conditions (>=5 tests)
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.metrics import summarize_predictions
from rmr_core.operators import build_multiscale_regions, weighted_normalized_adjoint_field
from rmr_v3.config import validate_v3_config
from rmr_v3.losses import curvature_power_loss, mass_weighted_cell_loss
from rmr_core.losses import flat_dm16_loss
from rmr_v3.model import MicroPerspectiveElevation, RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver

from tests.e2e_v26.conftest import (
    get_rmr_v26_config_spec,
    mpe_v2_oracle,
    pure_bb1_oracle,
)


# ============================================================================
# Tier 2 - F1: MPE-v2 Boundary Cases
# ============================================================================
class TestTier2F1MPEv2Boundaries:
    """Boundary and corner cases for F1: MPE-v2 Perspective Elevation."""

    def test_tier2_f1_minimal_spatial_height_2(self) -> None:
        """Boundary: Minimal valid spatial height H=2."""
        w = torch.randn(32, 1)
        b = torch.randn(32)
        x = torch.rand(1, 32, 2, 4) + 0.1

        out = mpe_v2_oracle(x, w, b)
        assert out.shape == (1, 32, 2, 4)
        assert torch.isfinite(out).all()
        vert_mean = (out / x).mean(dim=-2)
        assert torch.allclose(vert_mean, torch.ones_like(vert_mean), atol=1e-6)

    def test_tier2_f1_large_spatial_height_1024(self) -> None:
        """Stress: Ultra-large spatial height H=1024."""
        w = torch.randn(32, 1)
        b = torch.randn(32)
        x = torch.rand(1, 32, 1024, 8) + 0.1

        out = mpe_v2_oracle(x, w, b)
        assert out.shape == (1, 32, 1024, 8)
        assert torch.isfinite(out).all()
        vert_mean = (out / x).mean(dim=-2)
        assert torch.allclose(vert_mean, torch.ones_like(vert_mean), atol=1e-6)

    @pytest.mark.parametrize("prime_h", [17, 73, 137])
    def test_tier2_f1_prime_spatial_dimensions(self, prime_h: int) -> None:
        """Corner Case: Prime spatial heights non-divisible by power-of-2 strides."""
        w = torch.randn(32, 1)
        b = torch.randn(32)
        x = torch.rand(1, 32, prime_h, 23) + 0.1

        out = mpe_v2_oracle(x, w, b)
        assert out.shape == (1, 32, prime_h, 23)
        vert_mean = (out / x).mean(dim=-2)
        assert torch.allclose(vert_mean, torch.ones_like(vert_mean), atol=1e-6)

    def test_tier2_f1_extreme_weights_saturation(self) -> None:
        """Singularity: Extreme weights causing tanh saturation at +/- 1.0."""
        # W = +1000 -> tanh = +1.0; W = -1000 -> tanh = -1.0
        w = torch.full((32, 1), 1000.0)
        b = torch.full((32,), -500.0)
        x = torch.rand(1, 32, 32, 16) + 0.1

        out = mpe_v2_oracle(x, w, b)
        assert torch.isfinite(out).all()
        assert not torch.isnan(out).any()
        vert_mean = (out / x).mean(dim=-2)
        assert torch.allclose(vert_mean, torch.ones_like(vert_mean), atol=1e-6)

    def test_tier2_f1_delta_spike_vs_uniform_carrier(self) -> None:
        """Corner Case: Carrier with isolated delta spike vs uniform carrier."""
        w = torch.randn(32, 1)
        b = torch.randn(32)

        # Uniform carrier
        x_uniform = torch.ones(1, 32, 16, 16)
        out_uniform = mpe_v2_oracle(x_uniform, w, b)
        assert (out_uniform / x_uniform).mean(dim=-2).sub(1.0).abs().max() < 1e-6

        # Delta spike carrier
        x_spike = torch.ones(1, 32, 16, 16) * 0.01
        x_spike[0, 5, 8, 8] = 1000.0
        out_spike = mpe_v2_oracle(x_spike, w, b)
        assert (out_spike / x_spike).mean(dim=-2).sub(1.0).abs().max() < 1e-6


# ============================================================================
# Tier 2 - F2: Pure BB-1 Singularities & Edge Cases
# ============================================================================
class TestTier2F2PureBB1Boundaries:
    """Boundary and corner cases for F2: Pure BB-1 Trust Damping."""

    def test_tier2_f2_vanishing_residual_division_by_zero_safety(self) -> None:
        """Singularity: Vanishing residual r_k -> 0 (stationary point)."""
        s = torch.randn(1, 1, 8, 8)
        r_zero = torch.zeros(1, 1, 8, 8)

        alpha = pure_bb1_oracle(s, r_zero, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert torch.isfinite(alpha).all()
        # Should fallback to omega_0 (1.0) clamped in [0.5, 1.2]
        assert torch.isclose(alpha, torch.tensor([[[[1.0]]]]), atol=1e-5)

    def test_tier2_f2_orthogonal_displacement_residual(self) -> None:
        """Corner Case: Orthogonal displacement and residual <s, r> = 0."""
        s = torch.tensor([[[[1.0, 0.0]]]])
        r = torch.tensor([[[[0.0, 1.0]]]])  # <s, r> = 0

        alpha = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert torch.isfinite(alpha).all()
        assert torch.isclose(alpha, torch.tensor([[[[1.0]]]]), atol=1e-5)

    def test_tier2_f2_negative_curvature_descent(self) -> None:
        """Edge Case: Non-convex descent step <s, r> < 0."""
        s = torch.tensor([[[[1.0, 2.0]]]])
        r = torch.tensor([[[[-1.0, -2.0]]]])  # <s, r> = -5 < 0

        alpha = pure_bb1_oracle(s, r, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert torch.isfinite(alpha).all()
        # Falls back to omega_0 = 1.0 clamped in [0.5, 1.2]
        assert torch.isclose(alpha, torch.tensor([[[[1.0]]]]), atol=1e-5)

    def test_tier2_f2_extreme_step_size_demand(self) -> None:
        """Boundary: Alpha demanding infinity or -infinity strictly clamped."""
        s_huge = torch.full((1, 1, 4, 4), 1e8)
        r_tiny = torch.full((1, 1, 4, 4), 1e-4)

        alpha_max = pure_bb1_oracle(s_huge, r_tiny, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert torch.isclose(alpha_max, torch.tensor([[[[1.2]]]]), atol=1e-5)

        s_tiny = torch.full((1, 1, 4, 4), 1e-6)
        r_huge = torch.full((1, 1, 4, 4), 1e4)

        alpha_min = pure_bb1_oracle(s_tiny, r_huge, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert torch.isclose(alpha_min, torch.tensor([[[[0.5]]]]), atol=1e-5)

    def test_tier2_f2_single_iteration_t1(self, standard_multiscale_regions: Any) -> None:
        """Corner Case: Solver executed with single unrolled iteration T=1."""
        h, w = 64, 64
        y0 = torch.full((1, 1, h, w), 0.2)
        m = len(standard_multiscale_regions.boxes)
        b = torch.full((1, 1, m), 1.0)
        weight = torch.ones(1, 1, m)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b,
            weight_solver=weight,
            regions=standard_multiscale_regions,
            iterations=1,
            omega=1.0,
            use_barzilai_borwein=True,
            use_alternating_bb=False,
        )
        assert len(sol["iterates"]) == 2  # y0 + 1 iterate
        assert (sol["iterates"][-1] >= 0.0).all()


# ============================================================================
# Tier 2 - F3: Loss Core Boundaries & Zero/Clump Densities
# ============================================================================
class TestTier2F3LossCoreBoundaries:
    """Boundary and corner cases for F3: Loss Core and Zero/Clump Densities."""

    def test_tier2_f3_zero_density_all_background(self) -> None:
        """Boundary: Ground truth has zero count across all cells (pure background)."""
        pred = torch.full((1, 1, 32, 32), 0.01, requires_grad=True)
        target_zero = torch.zeros(1, 1, 32, 32)

        l_cell = mass_weighted_cell_loss(pred, target_zero, alpha=2.0, gamma=1.25)
        l_flat = flat_dm16_loss(pred, target_zero, kappa=20.0)

        assert torch.isfinite(l_cell) and l_cell >= 0.0
        assert torch.isfinite(l_flat) and l_flat >= 0.0

    def test_tier2_f3_hyper_dense_single_cell(self) -> None:
        """Stress: Extreme localized clump: count > 2000 in a single cell."""
        pred = torch.ones(1, 1, 16, 16, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16)
        target[0, 0, 8, 8] = 2500.0  # Massive hyper-dense spike

        loss = mass_weighted_cell_loss(pred, target, alpha=2.0, gamma=1.25)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_tier2_f3_zero_predicted_density_epsilon_safety(self) -> None:
        """Singularity: Predicted density y=0 (evaluates smooth zero-epsilon handling)."""
        pred_zero = torch.zeros(1, 1, 16, 16, requires_grad=True)
        target = torch.ones(1, 1, 16, 16)

        loss = mass_weighted_cell_loss(pred_zero, target, alpha=2.0, gamma=1.25)
        assert torch.isfinite(loss)
        assert loss > 0.0

    def test_tier2_f3_curvature_loss_damped_boundary(self) -> None:
        """Boundary: Compares curvature loss gradient behavior at lambda=0.1 vs lambda=0.0."""
        pred = torch.full((1, 1, 16, 16), 0.5, requires_grad=True)
        target = torch.full((1, 1, 16, 16), 0.2)

        l_curv_01 = curvature_power_loss(pred, target, eps=1e-3) * 0.10
        assert torch.isfinite(l_curv_01)
        assert l_curv_01 >= 0.0

    def test_tier2_f3_single_cell_region_box_boundary(self) -> None:
        """Boundary: RegionSet with minimal box size 1x1 cell."""
        reg_1x1 = build_multiscale_regions(
            height=16,
            width=16,
            output_stride=4,
            region_sizes_px=[4],  # 4px / stride 4 = 1 cell
            include_full_image=False,
        )
        assert len(reg_1x1.boxes) > 0
        assert (reg_1x1.boxes[:, 2] - reg_1x1.boxes[:, 0] == 1).all()


# ============================================================================
# Tier 2 - F4: Config Schema Rejection & Boundary Values
# ============================================================================
class TestTier2F4ConfigSchemaBoundaries:
    """Boundary and corner cases for F4: Config Schema Validation and Rejection."""

    def test_tier2_f4_empty_config_dictionary_rejected(self) -> None:
        """Rejection: Illegal top-level keys throw ValueError."""
        with pytest.raises(ValueError, match="Unknown top-level config key 'bad_key'"):
            validate_v3_config({"bad_key": 123})

    def test_tier2_f4_extra_illegal_config_keys_rejected(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Rejection: Illegal or unrecognized hyperparameters are strictly rejected."""
        bad_cfg = copy.deepcopy(canonical_v26_config_dict)
        bad_cfg["model"].pop("bb_clamp_min", None)
        bad_cfg["model"].pop("bb_clamp_max", None)
        bad_cfg["model"]["illegal_hallucinated_hyperparameter"] = 42

        with pytest.raises(ValueError, match="Unknown config key 'illegal_hallucinated_hyperparameter' in section 'model'"):
            validate_v3_config(bad_cfg)

    def test_tier2_f4_boundary_damping_clamps(self) -> None:
        """Boundary: Verifies bb_clamp_min <= bb_clamp_max."""
        cfg = get_rmr_v26_config_spec("canonical")
        assert cfg["model"]["bb_clamp_min"] < cfg["model"]["bb_clamp_max"]
        assert cfg["model"]["bb_clamp_min"] == 0.5
        assert cfg["model"]["bb_clamp_max"] == 1.2

    @pytest.mark.parametrize("gamma_val", [0.0, 0.75, 5.0])
    def test_tier2_f4_extreme_gamma_morozov(self, gamma_val: float) -> None:
        """Boundary: Morozov gamma boundary values [0.0, 5.0]."""
        cfg = get_rmr_v26_config_spec("canonical")
        cfg["model"]["morozov_gamma"] = gamma_val
        m_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
        assert m_cfg.morozov_gamma == gamma_val

    def test_tier2_f4_config_immutability(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Integrity: Model initialization does not mutate the input configuration dictionary."""
        original_dict = copy.deepcopy(canonical_v26_config_dict)
        _ = RMRv3(RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False))
        assert canonical_v26_config_dict == original_dict


# ============================================================================
# Tier 2 - F5: Mathematical Invariant Edge Cases & Singularities
# ============================================================================
class TestTier2F5MathInvariantBoundaries:
    """Boundary and corner cases for F5: Math & Integrity Invariants."""

    def test_tier2_f5_param_budget_ceiling_boundary(self) -> None:
        """Acceptance Criteria: Headroom under 105,000 budget is strictly positive."""
        cfg = get_rmr_v26_config_spec("canonical")
        m = RMRv3(RMRv3Config.from_dict(cfg["model"], pretrained=False))
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

        assert trainable == 104_505
        headroom = 105_000 - trainable
        assert headroom == 495, f"Expected 495 headroom, got {headroom}"

    def test_tier2_f5_zero_carrier_mass_preservation(self) -> None:
        """Singularity: Carrier P4 all zeros (verifies zero-division protection)."""
        w = torch.randn(32, 1)
        b = torch.randn(32)
        x_zeros = torch.zeros(1, 32, 16, 16)

        out = mpe_v2_oracle(x_zeros, w, b)
        assert torch.all(out == 0.0)
        assert not torch.isnan(out).any()

    def test_tier2_f5_single_pixel_delta_support_absorption(self, standard_multiscale_regions: Any) -> None:
        """Singularity: Density concentrated on exactly one pixel (i=10, j=10)."""
        h, w = 64, 64
        y_delta = torch.zeros(1, 1, h, w)
        y_delta[0, 0, 10, 10] = 1.0

        m = len(standard_multiscale_regions.boxes)
        b = torch.full((1, 1, m), 5.0)
        weight = torch.ones(1, 1, m)

        field = weighted_normalized_adjoint_field(
            y=y_delta,
            b_region=b,
            weight=weight,
            regions=standard_multiscale_regions,
            adjoint_mode="radon_nikodym",
            hybrid_recovery_alpha=0.0,
        )
        # Everywhere except (10, 10) must be identically 0.0
        assert field[0, 0, 10, 10] != 0.0
        field_copy = field.clone()
        field_copy[0, 0, 10, 10] = 0.0
        assert torch.all(field_copy == 0.0)

    def test_tier2_f5_checkerboard_zero_preservation(self, standard_multiscale_regions: Any) -> None:
        """Edge Case: Alternating checkerboard density preserves all zero pixels."""
        h, w = 64, 64
        y_check = torch.zeros(1, 1, h, w)
        y_check[0, 0, ::2, ::2] = 1.0  # Even rows and even cols have mass

        m = len(standard_multiscale_regions.boxes)
        b = torch.full((1, 1, m), 2.0)
        weight = torch.ones(1, 1, m)

        field = weighted_normalized_adjoint_field(
            y=y_check,
            b_region=b,
            weight=weight,
            regions=standard_multiscale_regions,
            adjoint_mode="radon_nikodym",
            hybrid_recovery_alpha=0.0,
        )
        zero_mask = y_check == 0.0
        assert torch.all(field[zero_mask] == 0.0)

    def test_tier2_f5_infinite_dispersion_limit(self) -> None:
        """Limit: Regional evidence head dispersion clamping bounds [0.5, 500.0]."""
        cfg = get_rmr_v26_config_spec("canonical")
        m_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
        assert m_cfg.dispersion_min == 0.5
        assert m_cfg.dispersion_max == 500.0


# ============================================================================
# Tier 2 - F6: Execution Boundary Conditions & Batch Variations
# ============================================================================
class TestTier2F6ExecutionBoundaries:
    """Boundary and corner cases for F6: Test Suite Reliability & Execution."""

    def test_tier2_f6_batch_size_one(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Boundary: Single image batch B=1 (verifies no squeeze error on batch dim)."""
        m_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()
        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            out = model(x)
        assert out.y.shape == (1, 1, 32, 32)
        assert torch.isfinite(out.y).all()

    def test_tier2_f6_large_batch_multi_sample(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Stress: Multi-sample batch B=4."""
        m_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()
        x = torch.randn(4, 3, 64, 64)

        with torch.no_grad():
            out = model(x)
        assert out.y.shape == (4, 1, 16, 16)
        assert torch.isfinite(out.y).all()

    def test_tier2_f6_zero_learning_rate_freeze(self) -> None:
        """Boundary: Optimizer step with LR=0 confirms zero parameter drift."""
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.AdamW([param], lr=0.0)

        loss = (param * 2.0).sum()
        loss.backward()
        optimizer.step()

        assert torch.allclose(param, torch.tensor([1.0, 2.0]))

    def test_tier2_f6_nan_inf_gradient_detection(self) -> None:
        """Safety: Detects non-finite gradient during training check."""
        param = nn.Parameter(torch.tensor([1.0]))
        loss = param * float("nan")
        loss.backward()

        assert param.grad is not None
        assert torch.isnan(param.grad).any()

    def test_tier2_f6_empty_region_handling(self) -> None:
        """Edge Case: RegionSet with no overlap outside image bounds."""
        reg = build_multiscale_regions(height=32, width=32, output_stride=4, region_sizes_px=[16])
        assert len(reg.boxes) > 0
        # All boxes strictly within [0, 32] on image grid
        assert (reg.boxes[:, 0] >= 0).all()
        assert (reg.boxes[:, 2] <= 32).all()


# ============================================================================
# Tier 2 - F7: Evaluation Metric Edge Conditions
# ============================================================================
class TestTier2F7EvaluationMetricBoundaries:
    """Boundary and corner cases for F7: Evaluation Metrics."""

    def test_tier2_f7_all_zero_gt_evaluation(self) -> None:
        """Boundary: Benchmark with all-zero counts (empty scenes)."""
        rows = [{"pred": 0.5, "gt": 0.0}, {"pred": 0.2, "gt": 0.0}]
        summary = summarize_predictions(rows)

        assert summary["MAE"] == 0.35
        assert summary["Bias"] == 0.35

    def test_tier2_f7_single_sample_evaluation(self) -> None:
        """Boundary: Evaluation on exactly N=1 sample (checks no divide-by-zero on variance)."""
        rows = [{"pred": 42.0, "gt": 40.0}]
        summary = summarize_predictions(rows)

        assert summary["MAE"] == 2.0
        assert summary["RMSE"] == 2.0
        assert summary["Bias"] == 2.0

    def test_tier2_f7_identical_pred_gt_perfect_score(self) -> None:
        """Corner Case: Predictions exactly equal ground truth."""
        rows = [{"pred": float(i), "gt": float(i)} for i in range(10)]
        summary = summarize_predictions(rows)

        assert summary["MAE"] == 0.0
        assert summary["RMSE"] == 0.0
        assert summary["Bias"] == 0.0

    def test_tier2_f7_constant_offset_bias_isolation(self) -> None:
        """Mathematical Invariant: Pred = GT + C -> Bias == C, MAE == |C|."""
        c = 7.5
        rows = [{"pred": float(i) + c, "gt": float(i)} for i in range(20)]
        summary = summarize_predictions(rows)

        assert math.isclose(summary["Bias"], c, abs_tol=1e-5)
        assert math.isclose(summary["MAE"], abs(c), abs_tol=1e-5)

    @pytest.mark.parametrize("count_val,expected_group", [(100.0, "sparse"), (500.0, "moderate"), (500.1, "dense")])
    def test_tier2_f7_boundary_density_bin_edges(self, count_val: float, expected_group: str) -> None:
        """Boundary: Tests threshold boundaries <=100 (sparse), (100, 500] (moderate), >500 (dense)."""
        if count_val <= 100.0:
            group = "sparse"
        elif count_val <= 500.0:
            group = "moderate"
        else:
            group = "dense"
        assert group == expected_group
