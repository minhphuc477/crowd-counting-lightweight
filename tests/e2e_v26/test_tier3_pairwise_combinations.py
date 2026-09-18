"""Tier 3: Cross-Feature Combinations Test Suite for RMR-v26.

Verifies pairwise feature interactions and cross-cutting dynamics:
- F1 (MPE-v2) x F2 (BB-1 Solver)
- F1 (MPE-v2) x F3 (Convex Loss Triad)
- F2 (BB-1 Solver) x F3 (Convex Loss Backward Pass)
- F4 (6-Model Configs) x F5 (Parameter Budgets & Headroom)
- F1 (MPE-v2) x F5 (Mass Conservation Invariance after Parameter Updates)
- F2 (BB-1 Solver) x F5 (Support Absorption Preservation across All Iterates)
- F6 (E2E Pipeline) x F7 (Evaluation Metrics Aggregation)
- F4 (Ablation Matrix) x F4 (Clean Pairwise Hyperparameter Isolation)
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.heads import FineMeasureHead
from rmr_core.metrics import summarize_predictions
from rmr_core.operators import build_multiscale_regions
from rmr_v3.losses import mass_weighted_cell_loss, scale_balanced_regional_nb_nll
from rmr_core.losses import flat_dm16_loss
from rmr_v3.model import MicroPerspectiveElevation, RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver

from tests.e2e_v26.conftest import (
    get_rmr_v26_config_spec,
    mpe_v2_oracle,
    pure_bb1_oracle,
)


class TestTier3CrossFeatureCombinations:
    """Pairwise cross-feature integration test suite."""

    def test_tier3_f1_mpe_v2_with_f2_bb1_unrolled_solver(self, standard_multiscale_regions: Any) -> None:
        """Integration F1 x F2: MPE-v2 carrier elevation feeding into unrolled SIRT solver."""
        torch.manual_seed(42)
        h, w = 64, 64
        channels = 32

        # 1. Generate carrier features and apply MPE-v2 elevation
        w_elev = torch.randn(channels, 1) * 0.1
        b_elev = torch.randn(channels) * 0.1
        p4 = torch.rand(1, channels, h, w) + 0.1
        p4_tilde = mpe_v2_oracle(p4, w_elev, b_elev)

        # 2. Project P4_tilde to initial continuous measure y0 via FineMeasureHead
        head = FineMeasureHead(width=channels)
        z0 = head(p4_tilde)
        y0 = head.activate(z0)
        assert torch.isfinite(y0).all()
        assert (y0 >= 0.0).all()

        # 3. Pass y0 to unrolled SIRT solver with pure BB-1 step size
        m = len(standard_multiscale_regions.boxes)
        b_solver = torch.full((1, 1, m), 2.0)
        weight_solver = torch.ones(1, 1, m)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=standard_multiscale_regions,
            iterations=6,
            omega=1.0,
            use_barzilai_borwein=True,
            use_alternating_bb=False,
        )

        y_final = sol["iterates"][-1]
        assert y_final.shape == (1, 1, h, w)
        assert torch.isfinite(y_final).all()
        assert (y_final >= 0.0).all()

    def test_tier3_f1_mpe_v2_with_f3_convex_loss_triad(self) -> None:
        """Integration F1 x F3: MPE-v2 elevation supervised by convex loss triad."""
        channels = 32
        w_elev = nn.Parameter(torch.randn(channels, 1) * 0.05)
        b_elev = nn.Parameter(torch.randn(channels) * 0.05)
        p4 = torch.rand(1, channels, 32, 32, requires_grad=True)

        p4_tilde = mpe_v2_oracle(p4, w_elev, b_elev)
        head = FineMeasureHead(width=channels)
        y_pred = head(p4_tilde)

        y_target = torch.rand(1, 1, 32, 32)
        l_cell = mass_weighted_cell_loss(y_pred, y_target, alpha=2.0, gamma=1.25)
        l_flat = flat_dm16_loss(y_pred, y_target, kappa=20.0)
        total_loss = 0.5 * l_cell + 1.0 * l_flat

        total_loss.backward()
        assert w_elev.grad is not None and torch.isfinite(w_elev.grad).all()
        assert b_elev.grad is not None and torch.isfinite(b_elev.grad).all()
        assert p4.grad is not None and torch.isfinite(p4.grad).all()

    def test_tier3_f2_bb1_solver_with_f3_composite_loss(self, standard_multiscale_regions: Any) -> None:
        """Integration F2 x F3: Backpropagating composite loss through all 6 unrolled iterations."""
        h, w = 32, 32
        y0 = torch.full((1, 1, h, w), 0.25, requires_grad=True)
        m = len(standard_multiscale_regions.boxes)
        b_solver = torch.full((1, 1, m), 1.0)
        weight = torch.ones(1, 1, m)

        # Reg with appropriate size for 32x32
        reg_32 = build_multiscale_regions(height=32, width=32, output_stride=4, region_sizes_px=[16, 32])
        m_32 = len(reg_32.boxes)
        b_sol_32 = torch.full((1, 1, m_32), 1.0)
        w_sol_32 = torch.ones(1, 1, m_32)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_sol_32,
            weight_solver=w_sol_32,
            regions=reg_32,
            iterations=6,
            omega=1.0,
            use_barzilai_borwein=True,
            use_alternating_bb=False,
        )

        y_final = sol["iterates"][-1]
        y_gt = torch.zeros(1, 1, h, w)
        loss = mass_weighted_cell_loss(y_final, y_gt, alpha=2.0, gamma=1.25)
        loss.backward()

        assert y0.grad is not None
        assert torch.isfinite(y0.grad).all()
        assert y0.grad.abs().sum().item() > 0.0

    def test_tier3_f4_configs_with_f5_param_budget_across_all_6_models(self) -> None:
        """Integration F4 x F5: Parameter budget validation across the entire 6-model matrix."""
        variants = [
            ("canonical", 104_505),
            ("ablation_no_elevation", 104_441),
            ("ablation_no_bb", 104_505),
            ("ablation_no_morozov", 104_505),
            ("ablation_with_curv01", 104_505),
            ("control_no_solver", 104_505),
        ]
        param_counts = {}
        for var_name, expected_p in variants:
            cfg = get_rmr_v26_config_spec(var_name)
            model = RMRv3(RMRv3Config.from_dict(cfg["model"], pretrained=False))
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts[var_name] = num_params
            assert num_params <= 105_000, f"{var_name} exceeded budget"
            assert num_params == expected_p, f"{var_name} parameter mismatch"

        # Exact delta between canonical and no_elevation must be exactly 64 (MPE-v2 parameters)
        delta_elevation = param_counts["canonical"] - param_counts["ablation_no_elevation"]
        assert delta_elevation == 64, f"Expected 64 parameter diff, got {delta_elevation}"

    def test_tier3_f1_mpe_v2_with_f5_mass_conservation_after_weight_updates(self) -> None:
        """Integration F1 x F5: Mass conservation invariant strictly holds after SGD/AdamW parameter updates."""
        channels = 32
        w = nn.Parameter(torch.zeros(channels, 1))
        b = nn.Parameter(torch.zeros(channels))
        optimizer = torch.optim.AdamW([w, b], lr=1e-2)

        # Simulate 10 gradient optimization steps
        for step in range(10):
            optimizer.zero_grad()
            x = torch.rand(2, channels, 32, 16) + 0.1
            out = mpe_v2_oracle(x, w, b)
            loss = out.sum()
            loss.backward()
            optimizer.step()

            # Mass conservation MUST hold identically after every parameter step
            vert_mean = (out / x).mean(dim=-2)
            diff = (vert_mean - 1.0).abs().max().item()
            assert diff < 1e-6, f"Mass conservation violated at step {step}: diff={diff}"

    def test_tier3_f2_bb1_solver_with_f5_support_absorption(self, standard_multiscale_regions: Any) -> None:
        """Integration F2 x F5: Multi-step unrolled solver strictly preserves Radon-Nikodym support absorption."""
        h, w = 64, 64
        # Semi-plane density: left half positive, right half zero
        y0 = torch.zeros(1, 1, h, w)
        y0[0, 0, :, :32] = 0.5

        m = len(standard_multiscale_regions.boxes)
        b_solver = torch.full((1, 1, m), 10.0)
        weight = torch.ones(1, 1, m)

        sol = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight,
            regions=standard_multiscale_regions,
            iterations=6,
            omega=1.0,
            adjoint_mode="radon_nikodym",
            use_barzilai_borwein=True,
            use_alternating_bb=False,
        )

        zero_mask = y0 == 0.0
        for it_idx, iterate in enumerate(sol["iterates"]):
            leaked = iterate[zero_mask].abs().max().item()
            assert leaked == 0.0, f"Support absorption leaked at iterate {it_idx}: max={leaked}"

    def test_tier3_f6_model_pipeline_with_f7_evaluation_metrics(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Integration F6 x F7: End-to-end forward inference mapped into prediction summary metrics."""
        m_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()

        synthetic_images = [torch.randn(1, 3, 64, 64) for _ in range(3)]
        ground_truth_counts = [15.0, 120.0, 550.0]

        predictions = []
        with torch.no_grad():
            for img in synthetic_images:
                out = model(img)
                pred_count = float(out.y.sum().item())
                predictions.append(pred_count)

        rows = [{"pred": p, "gt": g} for p, g in zip(predictions, ground_truth_counts)]
        summary = summarize_predictions(rows)

        assert "MAE" in summary
        assert "RMSE" in summary
        assert "Bias" in summary
        assert summary["MAE"] >= 0.0
        assert torch.isfinite(torch.tensor(summary["MAE"]))

    def test_tier3_f4_ablation_matrix_pairwise_isolation(self) -> None:
        """Integration F4 x F4: Pairwise diffs verify strictly 1 hyperparameter is altered per ablation."""
        canonical = get_rmr_v26_config_spec("canonical")

        # 1. no_elevation diff
        no_elev = get_rmr_v26_config_spec("ablation_no_elevation")
        assert no_elev["model"]["use_perspective_elevation"] is False
        assert canonical["model"]["use_perspective_elevation"] is True
        # All other model keys identical
        c_m = {k: v for k, v in canonical["model"].items() if k != "use_perspective_elevation"}
        n_m = {k: v for k, v in no_elev["model"].items() if k != "use_perspective_elevation"}
        assert c_m == n_m

        # 2. no_bb diff
        no_bb = get_rmr_v26_config_spec("ablation_no_bb")
        assert no_bb["model"]["use_barzilai_borwein"] is False
        assert canonical["model"]["use_barzilai_borwein"] is True
        c_m_bb = {k: v for k, v in canonical["model"].items() if k != "use_barzilai_borwein"}
        n_m_bb = {k: v for k, v in no_bb["model"].items() if k != "use_barzilai_borwein"}
        assert c_m_bb == n_m_bb

        # 3. no_morozov diff
        no_mor = get_rmr_v26_config_spec("ablation_no_morozov")
        assert no_mor["model"]["morozov_gamma"] == 0.0
        assert canonical["model"]["morozov_gamma"] == 0.75
        c_m_mor = {k: v for k, v in canonical["model"].items() if k != "morozov_gamma"}
        n_m_mor = {k: v for k, v in no_mor["model"].items() if k != "morozov_gamma"}
        assert c_m_mor == n_m_mor

        # 4. with_curv01 diff
        curv01 = get_rmr_v26_config_spec("ablation_with_curv01")
        assert curv01["loss"]["lambda_curvature"] == 0.10
        assert canonical["loss"]["lambda_curvature"] == 0.0
        assert canonical["model"] == curv01["model"]

        # 5. control_no_solver diff
        no_sol = get_rmr_v26_config_spec("control_no_solver")
        assert no_sol["model"]["enable_solver"] is False
        assert canonical["model"]["enable_solver"] is True
        c_m_sol = {k: v for k, v in canonical["model"].items() if k != "enable_solver"}
        n_m_sol = {k: v for k, v in no_sol["model"].items() if k != "enable_solver"}
        assert c_m_sol == n_m_sol
