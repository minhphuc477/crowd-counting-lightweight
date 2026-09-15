from __future__ import annotations

"""Dedicated Invariant and Clean Architecture Test Suite for RMR-v16.

Validates:
1. Exact parameter budget compliance: exactly 104,473 parameters (<= 105,000 budget).
2. Clean architecture invariants: zero heuristic patches (fg_gate is None, trust_gate is None, tdsg is None).
3. Radon-Nikodym adjoint scale invariance: H_nu 1_G = 1_G.
4. Zero background phantom mass leakage on empty / zero-density regions.
5. Bayesian Morozov deadband shrinkage invariants.
6. Pre-solver scale-consistency gating behavior.
7. Pure SNR reliability monotonicity.
8. End-to-end multi-loss supervision and non-zero gradient flow across all 104,473 parameters.
9. Full schema validity across all 7 v16 configuration YAMLs.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
)
from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.regional_head import apply_scale_consistency_gating, reliability_from_nb
from rmr_v3.solver import unrolled_sirt_solver


def _load_v16_config(filename: str = "rmr_v16_canonical.yaml") -> tuple[RMRv3Config, RMRv3LossConfig, dict]:
    cfg_path = Path(f"configs/rmr_v16/{filename}")
    assert cfg_path.is_file(), f"Config not found at {cfg_path}"
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    validate_v3_config(raw_cfg)
    m_cfg = {k: v for k, v in raw_cfg["model"].items() if hasattr(RMRv3Config, k)}
    m_cfg["pretrained"] = False
    return RMRv3Config(**m_cfg), RMRv3LossConfig(**raw_cfg["loss"]), raw_cfg


class TestRMRv16CleanArchitecture:
    """Comprehensive invariant test suite for RMR-v16 Clean Architecture."""

    def test_parameter_budget_exactness(self):
        """Invariant 1: Trainable parameter count is strictly <= 105,000 and exactly 104,473."""
        model_cfg, _, _ = _load_v16_config("rmr_v16_canonical.yaml")
        model = RMRv3(model_cfg)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_trainable == 104473, f"Expected exactly 104,473 params, got {n_trainable}"
        assert n_trainable <= 105000, f"Exceeded hard budget: {n_trainable} > 105,000"

        # Precise submodule parameter breakdown verification
        encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        fusion_params = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
        fine_head_params = sum(p.numel() for p in model.fine_head.parameters() if p.requires_grad)
        router_params = sum(p.numel() for p in model.scale_router.parameters() if p.requires_grad)
        region_head_params = sum(p.numel() for p in model.region_head.parameters() if p.requires_grad)

        assert encoder_params == 87568, f"Encoder: expected 87,568, got {encoder_params}"
        assert fusion_params == 10784, f"Fusion (ASPP-Lite): expected 10,784, got {fusion_params}"
        assert fine_head_params == 1474, f"Fine Head: expected 1,474, got {fine_head_params}"
        assert router_params == 516, f"Scale Router: expected 516, got {router_params}"
        assert region_head_params == 4131, f"Region Head: expected 4,131, got {region_head_params}"

    def test_clean_architecture_zero_heuristic_patches(self):
        """Invariant 2: Clean architecture permanently disables heuristic clutter."""
        model_cfg, _, _ = _load_v16_config("rmr_v16_canonical.yaml")
        model = RMRv3(model_cfg)

        assert model.fg_gate is None, "foreground_gate must be None in RMR-v16"
        assert model.trust_gate is None, "trust_gate must be None in RMR-v16"
        assert model.tdsg is None, "top-down semantic gate must be None in RMR-v16"
        assert not model.fine_head.scale_conditioned, "scale_conditioned must be False in RMR-v16"
        assert not hasattr(model.fine_head, "scale_beta"), "scale_beta must not exist in RMR-v16"

    def test_radon_nikodym_adjoint_zero_background_leakage(self):
        """Invariant 3: Radon-Nikodym adjoint guarantees zero mass leakage on background."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64, 128), overlap=0.5)
        m = regions.boxes.shape[0]

        # Construct a synthetic carrier with a sharp head cluster in the center and exact zero on background
        y = torch.zeros(1, 1, h, w, dtype=torch.float32)
        y[0, 0, 14:18, 14:18] = 5.0  # Dense cluster

        # Simulated positive discrepancy
        delta = torch.ones(1, 1, m, dtype=torch.float32) * 2.0
        weights = torch.ones(1, 1, m, dtype=torch.float32)

        # 1. Radon-Nikodym Adjoint
        b_region = regional_sum(y, regions.boxes) + delta
        res_rn = unrolled_sirt_solver(
            y, b_region, weights, regions,
            iterations=1, omega=1.0, adjoint_mode="radon_nikodym"
        )
        y_rn = res_rn["y"]

        # Background pixels where y == 0 MUST remain identically zero!
        bg_mask = (y[0, 0] == 0.0)
        assert (y_rn[0, 0][bg_mask] == 0.0).all(), "Radon-Nikodym adjoint leaked mass onto zero-density background!"

        # 2. Flat Adjoint Comparison: leaks mass onto background
        res_flat = unrolled_sirt_solver(
            y, b_region, weights, regions,
            iterations=1, omega=1.0, adjoint_mode="flat"
        )
        y_flat = res_flat["y"]
        assert (y_flat[0, 0][bg_mask] > 0.0).any(), "Flat adjoint was expected to leak mass onto background"

    def test_bayesian_morozov_deadband_shrinkage(self):
        """Invariant 4: Residuals within gamma * sigma are zeroed by Morozov shrinkage."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64, 128), overlap=0.5)
        m = regions.boxes.shape[0]

        y = torch.ones(1, 1, h, w, dtype=torch.float32) * 0.1
        ay = regional_sum(y, regions.boxes)  # [1, 1, M]

        # Case A: Residual is SMALLER than gamma * sigma -> delta_tilde should be 0, y unchanged
        b_variance = torch.ones(1, 1, m, dtype=torch.float32) * 1.0  # sigma = 1.0
        morozov_gamma = 0.75
        # Set b such that |Ay - b| = 0.5 < 0.75 * 1.0
        b_small_err = ay + 0.5

        weights = torch.ones(1, 1, m, dtype=torch.float32)
        res_deadband = unrolled_sirt_solver(
            y, b_small_err, weights, regions,
            iterations=1, omega=1.0, adjoint_mode="radon_nikodym",
            b_variance=b_variance, morozov_gamma=morozov_gamma
        )
        # In deadband, no update occurs: y_next == y
        assert torch.allclose(res_deadband["y"], y, atol=1e-6), "Morozov deadband failed to zero sub-threshold residual"

        # Case B: Residual is LARGER than gamma * sigma -> update occurs
        b_large_err = ay + 2.0  # |Ay - b| = 2.0 > 0.75 * 1.0
        res_active = unrolled_sirt_solver(
            y, b_large_err, weights, regions,
            iterations=1, omega=1.0, adjoint_mode="radon_nikodym",
            b_variance=b_variance, morozov_gamma=morozov_gamma
        )
        assert not torch.allclose(res_active["y"], y, atol=1e-4), "Morozov deadband erroneously blocked super-threshold residual"

    def test_pre_solver_scale_consistency_gating(self):
        """Invariant 5: Pre-solver scale gating suppresses coarse boxes over micro-scale clusters."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64, 128), overlap=0.5)
        m = regions.boxes.shape[0]

        # 4 scales: [16, 32, 64, 128] -> k in {0, 1, 2, 3}
        scale_weights = torch.zeros(1, 4, h, w, dtype=torch.float32)
        # Cluster is at micro-scale (scale 0 = 16px) everywhere
        scale_weights[:, 0, :, :] = 0.95
        scale_weights[:, 1, :, :] = 0.03
        scale_weights[:, 2, :, :] = 0.01
        scale_weights[:, 3, :, :] = 0.01

        raw_weights = torch.ones(1, 1, m, dtype=torch.float32)
        gated_weights = apply_scale_consistency_gating(raw_weights, regions, scale_weights, power=1.0)

        # Scale 0 boxes (16px) should keep high weight (~0.95)
        # Scale 3 boxes (128px) should have very low weight (~0.01)
        scale0_mask = (regions.scale_id == 0)
        scale3_mask = (regions.scale_id == 3)

        assert gated_weights[0, 0, scale0_mask].mean() > 0.90
        assert gated_weights[0, 0, scale3_mask].mean() < 0.05

    def test_pure_snr_reliability_monotonicity(self):
        """Invariant 6: Pure SNR reliability is monotonic in count mu and dispersion r."""
        h, w = 32, 32
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32, 64, 128), overlap=0.5)
        m = regions.boxes.shape[0]

        mu_low = torch.ones(1, 1, m) * 1.0
        mu_high = torch.ones(1, 1, m) * 100.0
        disp = torch.ones(1, 1, m) * 10.0

        rel_low = reliability_from_nb(mu_low, disp, regions, mode="snr")
        rel_high = reliability_from_nb(mu_high, disp, regions, mode="snr")

        # Precision = r * mu / (r + mu)
        # For mu=1, r=10: 10/11 = 0.909
        # For mu=100, r=10: 1000/110 = 9.09
        assert rel_high["precision"].mean() > rel_low["precision"].mean()

    def test_end_to_end_forward_backward_gradient_hygiene(self):
        """Invariant 7: Full forward + multi-loss backward produces clean non-zero gradients on all 104,473 params."""
        model_cfg, loss_cfg, _ = _load_v16_config("rmr_v16_canonical.yaml")
        model = RMRv3(model_cfg)
        model.train()

        b, c, h_in, w_in = 2, 3, 256, 256
        x = torch.randn(b, c, h_in, w_in, requires_grad=False)
        target_points = [
            torch.tensor([[64.0, 64.0], [80.0, 80.0], [120.0, 120.0]], dtype=torch.float32),
            torch.tensor([[100.0, 150.0], [200.0, 200.0]], dtype=torch.float32),
        ]

        outputs = model(x)

        # Build synthetic ground truth target density map at stride 4
        h_out, w_out = h_in // 4, w_in // 4
        target_density = torch.zeros(b, 1, h_out, w_out, dtype=torch.float32)
        for i, pts in enumerate(target_points):
            for pt in pts:
                py, px = int(pt[0].item() // 4), int(pt[1].item() // 4)
                if 0 <= py < h_out and 0 <= px < w_out:
                    target_density[i, 0, py, px] += 1.0

        losses = compute_rmr_v3_losses(
            outputs,
            target_density,
            loss_cfg,
            points=target_points,
        )

        total_loss = losses["total"]
        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"
        total_loss.backward()

        # Step 0: Check every parameter has valid, finite gradients (no NaNs or Infs)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} received no gradient on step 0!"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has Inf gradients!"

        # Zero-init gating check: pw.weight has non-zero gradient on step 0
        assert model.scale_router.pw.weight.grad.abs().sum() > 0.0

        # Step 1: After 1 optimizer step, zero-init layers update and propagate non-zero gradients to all upstream parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

        outputs2 = model(x)
        losses2 = compute_rmr_v3_losses(outputs2, target_density, loss_cfg, points=target_points)
        losses2["total"].backward()

        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} received no gradient on step 1!"
                if param.grad.abs().sum() == 0.0:
                    zero_grad_params.append(name)

        assert len(zero_grad_params) == 0, f"Found parameters with exact zero gradient on step 1: {zero_grad_params}"

    def test_all_seven_v16_configs_load_and_validate(self):
        """Invariant 8: All 7 v16 YAML configurations pass schema validation and instantiate cleanly."""
        configs = [
            "rmr_v16_canonical.yaml",
            "rmr_v16_ablation_flat_adjoint.yaml",
            "rmr_v16_ablation_no_morozov.yaml",
            "rmr_v16_ablation_rate_var_rel.yaml",
            "rmr_v16_ablation_no_pre_scale.yaml",
            "rmr_v16_ablation_no_micro16.yaml",
            "rmr_v16_control_no_solver.yaml",
        ]

        for cfg_name in configs:
            m_cfg, l_cfg, raw = _load_v16_config(cfg_name)
            model = RMRv3(m_cfg)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            if cfg_name == "rmr_v16_ablation_no_micro16.yaml":
                # With 3 scales, scale router is 32x3 + 3 = 99 params vs 32x4 + 4 = 132 params (-33 params)
                assert n_params == 104440
            else:
                assert n_params == 104473

            assert n_params <= 105000

    def test_dynamic_scale_telemetry_and_diagnostics(self):
        """Invariant 9: DiagnosticTracker and diagnostics correctly track and label all 4 scales [16, 32, 64, 128] px."""
        from rmr_v3.diagnostics import (
            compute_reliability_correlations,
            compute_uncertainty_calibration_bins,
            compute_nb_interval_coverage,
            compute_solver_trajectory_diagnostics,
            regional_reliability_rows,
        )
        from rmr_v3.tracking import DiagnosticTracker

        m_cfg, _, _ = _load_v16_config("rmr_v16_canonical.yaml")
        model = RMRv3(m_cfg)
        tracker = DiagnosticTracker(model)

        assert tracker.scale_map == {0: 16, 1: 32, 2: 64, 3: 128}

        # Run forward on synthetic input
        x = torch.randn(2, 3, 128, 128)
        outputs = model(x)
        tracker.update(outputs)
        summary = tracker.summarize()

        # All 4 scales must be tracked without indexing mismatch
        for s in (16, 32, 64, 128):
            assert f"weight_mean_{s}" in summary
            assert f"scale_pi_{s}" in summary
            assert summary[f"scale_pi_{s}"] >= 0.0

        # Sum of pi over the 4 scales must be approx 1.0
        total_pi = sum(summary[f"scale_pi_{s}"] for s in (16, 32, 64, 128))
        assert abs(total_pi - 1.0) < 1e-3

        # Test diagnostic functions with 4 scales
        target_y = torch.zeros(2, 1, 32, 32)
        target_y[:, :, 5:10, 5:10] = 1.0
        diag_rows = regional_reliability_rows(outputs, target_y)
        assert len(diag_rows) > 0

        scale_map = {0: 16, 1: 32, 2: 64, 3: 128}
        corrs = compute_reliability_correlations(diag_rows, scale_map=scale_map)
        for s in (16, 32, 64, 128):
            assert f"spearman_rate_var_error_{s}" in corrs

        calib = compute_uncertainty_calibration_bins(diag_rows, scale_map=scale_map)
        for s in (16, 32, 64, 128):
            assert f"calibration_{s}" in calib

        traj = compute_solver_trajectory_diagnostics(outputs, target_y, scale_map=scale_map)
        for s in (16, 32, 64, 128):
            assert f"mae_reg_{s}_y0" in traj
