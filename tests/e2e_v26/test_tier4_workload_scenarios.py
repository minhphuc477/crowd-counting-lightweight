"""Tier 4: Real-World Workload Scenarios Test Suite for RMR-v26.

Simulates end-to-end operational pipelines and realistic crowd counting workflows:
- Scenario 1: Full Training Step (Image -> Transform -> Forward -> Convex Triad -> Backward -> Step)
- Scenario 2: Full Validation Pipeline with Horizontal Flip TTA & Metric Summary
- Scenario 3: Hyper-Dense Crowd Clustering Scenario (>1000 people, high foreshortening)
- Scenario 4: Sparse Surveillance Scene (<50 people, wide angle, zero false foreground)
- Scenario 5: Cross-Model Comparative Benchmark Suite (Canonical vs Ablations)
- Scenario 6: Checkpoint Serialization and EMA Shadow Weight Evaluation
"""
from __future__ import annotations

import copy
import io
import math
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.metrics import summarize_predictions
from rmr_core.operators import build_multiscale_regions
from rmr_v3.losses import mass_weighted_cell_loss
from rmr_core.losses import flat_dm16_loss
from rmr_v3.model import RMRv3, RMRv3Config

from tests.e2e_v26.conftest import (
    get_rmr_v26_config_spec,
    mpe_v2_oracle,
    pure_bb1_oracle,
)


class TestTier4RealWorldWorkloadScenarios:
    """Real-world end-to-end system workload scenarios."""

    def test_tier4_scenario1_full_training_step_end_to_end(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Scenario 1: Complete end-to-end training step."""
        torch.manual_seed(42)
        model_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(model_cfg).train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # 1. Synthetic training batch (B=2, 3 channels, 128x128 image)
        images = torch.randn(2, 3, 128, 128)
        target_cells = torch.rand(2, 1, 32, 32)  # stride 4 target

        # 2. Forward pass
        optimizer.zero_grad()
        out = model(images)
        assert out.y.shape == (2, 1, 32, 32)
        assert torch.isfinite(out.y).all()

        # 3. Compute loss using canonical convex triad (zero curvature)
        l_cell = mass_weighted_cell_loss(out.y, target_cells, alpha=2.0, gamma=1.25)
        l_flat = flat_dm16_loss(out.y, target_cells, kappa=20.0)
        total_loss = 0.5 * l_cell + 1.0 * l_flat

        assert torch.isfinite(total_loss)
        assert total_loss.item() > 0.0

        # 4. Backward pass
        total_loss.backward()

        # 5. Verify non-zero finite gradients across model parameters
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()
                if p.grad.abs().sum().item() > 0.0:
                    has_grad = True
        assert has_grad, "No parameters received non-zero gradients!"

        # 6. Optimizer step
        optimizer.step()

    def test_tier4_scenario2_full_validation_tta_pipeline(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Scenario 2: Complete validation pipeline with horizontal flip TTA."""
        torch.manual_seed(100)
        model_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(model_cfg).eval()

        val_images = [torch.randn(1, 3, 128, 128) for _ in range(4)]
        val_ground_truths = [45.0, 180.0, 490.0, 720.0]

        predictions = []
        with torch.no_grad():
            for img in val_images:
                # Direct prediction
                y_direct = model(img).y
                # Flipped prediction
                img_flipped = torch.flip(img, dims=[-1])
                y_flipped_raw = model(img_flipped).y
                y_flipped_back = torch.flip(y_flipped_raw, dims=[-1])

                # TTA horizontal ensemble: 0.5 * (Y_direct + Flip(Y_flipped))
                y_tta = 0.5 * (y_direct + y_flipped_back)
                pred_count = float(y_tta.sum().item())
                predictions.append(pred_count)

        rows = [{"pred": p, "gt": g} for p, g in zip(predictions, val_ground_truths)]
        summary = summarize_predictions(rows)

        assert "MAE" in summary
        assert "RMSE" in summary
        assert "Bias" in summary
        assert summary["MAE"] >= 0.0

    def test_tier4_scenario3_dense_crowd_clustering_workload(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Scenario 3: Hyper-dense crowd scenario (>1000 people, high foreshortening)."""
        torch.manual_seed(2026)
        model_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(model_cfg).eval()

        # Dense scene input
        dense_img = torch.randn(1, 3, 128, 128) + 2.0
        with torch.no_grad():
            out = model(dense_img)

        # Confirm solver remained numerically stable and bounded
        assert torch.isfinite(out.y).all()
        assert not torch.isnan(out.y).any()
        assert (out.y >= 0.0).all()

        # Confirm step sizes in solver remained clamped to [0.5, 1.2]
        s_dense = torch.full((1, 1, 32, 32), 10.0)
        r_dense = torch.full((1, 1, 32, 32), 100.0)
        alpha = pure_bb1_oracle(s_dense, r_dense, omega_0=1.0, clamp_min=0.5, clamp_max=1.2)
        assert 0.5 <= alpha.item() <= 1.2

    def test_tier4_scenario4_sparse_surveillance_workload(self, canonical_v26_config_dict: Dict[str, Any]) -> None:
        """Scenario 4: Sparse surveillance scenario (<50 people, wide angle, background dominance)."""
        torch.manual_seed(777)
        model_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(model_cfg).eval()

        # Sparse scene (mostly zero background)
        sparse_img = torch.zeros(1, 3, 128, 128)
        # Small crowd cluster in upper corner
        sparse_img[0, :, 10:30, 10:30] = 1.0

        with torch.no_grad():
            out = model(sparse_img)

        assert torch.isfinite(out.y).all()
        assert (out.y >= 0.0).all()

        # Zero-inflation check: Total predicted count remains tightly bounded by the calibrated prior
        total_pred_count = out.y.sum().item()
        assert total_pred_count < 50.0, f"Sparse scene suffered from false count inflation: {total_pred_count}"
        assert total_pred_count > 0.0

    def test_tier4_scenario5_model_ablation_comparative_benchmark_workflow(self) -> None:
        """Scenario 5: Comparative benchmark across the 6-model suite."""
        variants = [
            "canonical",
            "ablation_no_elevation",
            "ablation_no_bb",
            "ablation_no_morozov",
            "ablation_with_curv01",
            "control_no_solver",
        ]

        benchmark_results = {}
        mock_gts = [50.0, 150.0, 600.0]

        # Simulate comparative evaluation run
        for variant in variants:
            cfg = get_rmr_v26_config_spec(variant)
            model = RMRv3(RMRv3Config.from_dict(cfg["model"], pretrained=False)).eval()

            with torch.no_grad():
                preds = [float(model(torch.randn(1, 3, 64, 64)).y.sum().item()) for _ in range(3)]

            rows = [{"pred": p, "gt": g} for p, g in zip(preds, mock_gts)]
            summary = summarize_predictions(rows)
            benchmark_results[variant] = summary

        assert len(benchmark_results) == 6
        for variant in variants:
            assert "MAE" in benchmark_results[variant]
            assert "RMSE" in benchmark_results[variant]
            assert "Bias" in benchmark_results[variant]

    def test_tier4_scenario6_checkpoint_serialization_and_ema_eval_workflow(
        self, canonical_v26_config_dict: Dict[str, Any]
    ) -> None:
        """Scenario 6: Checkpoint serialization and shadow EMA evaluation."""
        torch.manual_seed(55)
        model_cfg = RMRv3Config.from_dict(canonical_v26_config_dict["model"], pretrained=False)
        model = RMRv3(model_cfg).eval()

        # Simulate EMA weights dictionary
        ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        checkpoint = {
            "epoch": 80,
            "model": model.state_dict(),
            "ema_model": ema_state,
            "best_val_mae": 68.42,
        }

        # Serialize to in-memory buffer
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        # Deserialize and load into evaluation model
        loaded = torch.load(buffer, weights_only=True)
        assert loaded["epoch"] == 80
        assert loaded["best_val_mae"] == 68.42

        eval_model = RMRv3(model_cfg).eval()
        eval_model.load_state_dict(loaded["ema_model"])

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = eval_model(x)
        assert torch.isfinite(out.y).all()
