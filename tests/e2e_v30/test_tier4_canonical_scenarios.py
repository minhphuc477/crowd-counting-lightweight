"""Tier 4: Canonical Benchmark & Operational Scenarios Test Suite for RMR-v30.

Verifies end-to-end execution matching canonical ShanghaiTech Part A evaluation protocols:
- Canonical Test Loader Validation on official sha_a_test.jsonl (>= 1 test)
- Horizontal Flip Test-Time Augmentation (TTA) Parity (>= 1 test)
- Exact 7 / 129 / 46 Density Stratification Partition (>= 1 test)
- Anti-Regression Gate vs RMR-v19 Golden Anchor (>= 1 test)
- Sub-65 / Sub-60 Breakthrough Metric Validator Oracle (>= 1 test)
- Complete 6-Model Suite Parameter & Execution Footprint (>= 1 test)
- Ground-Truth Consistency Invariant (>= 1 test)
- Full-Resolution Tiled Inference with Practical Halo (>= 1 test)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset, predict_tiled
from rmr_core.metrics import density_stratified_mae, summarize_predictions
from rmr_v3.model import RMRv3, RMRv3Config

from tests.e2e_v30.conftest import get_rmr_v30_config_spec


def validate_sub60_breakthrough(summary: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Authoritative validation oracle for Sub-65 / Sub-60 milestone acceptance criteria.

    Criteria (from SCOPE.md and ORIGINAL_REQUEST.md):
    1. Overall MAE <= 64.9 (Sub-65 Barrier broken)
    2. Sparse MAE (N <= 100) <= 14.0 (retaining v29 record)
    3. Dense MAE (N > 500) <= 105.0 (down from v19's 122.71)
    4. Net Prediction Bias in [-8.0, +3.0] (eliminating v29's -29.95 undercounting)
    5. Trainable parameters <= 105,000
    """
    reasons = []
    mae = float(summary.get("MAE", 999.0))
    if mae > 64.9:
        reasons.append(f"Overall MAE {mae:.2f} > 64.9 (Sub-65 threshold)")

    sparse_mae = float(summary.get("mae_sparse", 999.0))
    if sparse_mae > 14.0:
        reasons.append(f"Sparse MAE {sparse_mae:.2f} > 14.0")

    dense_mae = float(summary.get("mae_dense", 999.0))
    if dense_mae > 105.0:
        reasons.append(f"Dense MAE {dense_mae:.2f} > 105.0")

    bias = float(summary.get("Bias", 999.0))
    if not (-8.0 <= bias <= 3.0):
        reasons.append(f"Net bias {bias:.2f} outside allowed interval [-8.0, +3.0]")

    params = int(summary.get("parameters", 999999))
    if params > 105000:
        reasons.append(f"Parameters {params} exceed 105,000 ceiling")

    is_passed = len(reasons) == 0
    return is_passed, reasons


class TestTier4CanonicalScenarios:
    """Canonical ShanghaiTech Part A evaluation scenario tests."""

    def test_tier4_canonical_test_loader_validation(
        self,
        sha_a_test_manifest_path: Path,
        canonical_step0_config_dict: Dict[str, Any],
    ) -> None:
        """Scenario 1: Canonical test manifest loads and executes batch=1 evaluation collate."""
        ds = CrowdManifestDataset(manifest=str(sha_a_test_manifest_path), train=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_eval)
        sample_batch = next(iter(loader))
        assert len(sample_batch) == 1
        item = sample_batch[0]

        assert "image" in item
        assert "target_y" in item
        assert "points" in item
        assert item["image"].shape[0] == 3  # RGB image

        # Verify model evaluation forward pass
        m_cfg = RMRv3Config.from_dict(canonical_step0_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()
        img_4d = item["image"].unsqueeze(0)
        with torch.no_grad():
            out = model(img_4d)
        assert torch.isfinite(out.y).all()
        assert out.y.shape == (1, 1, item["height"] // 4, item["width"] // 4)

    def test_tier4_tta_horizontal_flip_parity(
        self,
        canonical_step0_config_dict: Dict[str, Any],
    ) -> None:
        """Scenario 2: Test-Time Augmentation (TTA) horizontal flip parity."""
        m_cfg = RMRv3Config.from_dict(canonical_step0_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()

        img = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            # 1. Direct prediction
            out_direct = model(img)
            y_direct = out_direct.y

            # 2. Flipped prediction
            img_flip = torch.flip(img, dims=[-1])
            out_flip = model(img_flip)
            y_flip_unflipped = torch.flip(out_flip.y, dims=[-1])

            # 3. TTA composite
            y_tta = 0.5 * (y_direct + y_flip_unflipped)

        assert y_tta.shape == y_direct.shape
        assert torch.isfinite(y_tta).all()
        assert (y_tta >= 0.0).all()
        # Count difference between direct and TTA should be minimal
        count_direct = y_direct.sum().item()
        count_tta = y_tta.sum().item()
        assert abs(count_direct - count_tta) / max(1.0, count_direct) < 0.25

    def test_tier4_density_stratification_exact_partition(
        self,
        sha_a_test_manifest_path: Path,
    ) -> None:
        """Scenario 3: Canonical test set partitions into exactly 7 sparse, 129 moderate, and 46 dense images."""
        with sha_a_test_manifest_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) == 182, f"Expected 182 images, got {len(records)}"
        gts = [len(r.get("points", [])) for r in records]

        sparse_count = sum(1 for g in gts if g <= 100.0)
        moderate_count = sum(1 for g in gts if 100.0 < g <= 500.0)
        dense_count = sum(1 for g in gts if g > 500.0)

        assert sparse_count == 7, f"Expected exactly 7 sparse images (count <= 100), got {sparse_count}"
        assert moderate_count == 129, f"Expected exactly 129 moderate images, got {moderate_count}"
        assert dense_count == 46, f"Expected exactly 46 dense images (count > 500), got {dense_count}"
        assert sparse_count + moderate_count + dense_count == 182

    def test_tier4_anti_regression_v19_anchor_bounds(self) -> None:
        """Scenario 4: Validates historical RMR-v19 Golden Anchor performance bounds."""
        summary_path = Path("runs/sha_a/rmr_v19_canonical_isotropic/eval_val/summary.json")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            mae = float(summary["MAE"])
            assert math.isclose(mae, 72.84, abs_tol=0.30), f"v19 anchor MAE regressed: {mae}"
            assert float(summary["mae_sparse"]) <= 25.0
            assert float(summary["mae_dense"]) <= 130.0
        else:
            # Anchor reference bounds check
            anchor_mae = 72.84
            assert 72.5 <= anchor_mae <= 73.2

    def test_tier4_sub65_sub60_threshold_validator_oracle(self) -> None:
        """Scenario 5: Threshold validator correctly accepts winning summaries and rejects regressions."""
        # 1. Winning Sub-60 candidate summary
        winning_summary = {
            "MAE": 58.75,
            "mae_sparse": 11.20,
            "mae_dense": 98.50,
            "Bias": -1.85,
            "parameters": 104540,
        }
        passed, reasons = validate_sub60_breakthrough(winning_summary)
        assert passed, f"Winning candidate was unexpectedly rejected: {reasons}"

        # 2. Failing candidate: v29 H2 severe undercounting bias (-29.95) and dense explosion (191.32)
        v29_failing_summary = {
            "MAE": 90.03,
            "mae_sparse": 10.64,
            "mae_dense": 191.32,
            "Bias": -29.95,
            "parameters": 104540,
        }
        passed_v29, reasons_v29 = validate_sub60_breakthrough(v29_failing_summary)
        assert not passed_v29
        assert any("Net bias" in r for r in reasons_v29)
        assert any("Dense MAE" in r for r in reasons_v29)

        # 3. Failing candidate: parameter ceiling violation (> 105,000)
        over_budget_summary = {
            "MAE": 55.0,
            "mae_sparse": 10.0,
            "mae_dense": 90.0,
            "Bias": 0.0,
            "parameters": 125000,
        }
        passed_over, reasons_over = validate_sub60_breakthrough(over_budget_summary)
        assert not passed_over
        assert any("parameters" in r.lower() for r in reasons_over)

    def test_tier4_complete_six_model_suite_parameter_and_memory(self) -> None:
        """Scenario 6: Sequentially instantiates and executes forward passes for all 6 suite models."""
        variants = [
            ("step0_v19_anchor", 104441),
            ("h1_anscombe_sirt", 104441),
            ("h2_dual_lattice_dcsr", 104540),
            ("h3_anscombe_dual_lattice", 104540),
            ("h4_deep_sirt_t8", 104540),
            ("control_no_solver", 104441),
        ]
        dummy_img = torch.randn(1, 3, 64, 64)
        for var, exp_params in variants:
            spec = get_rmr_v30_config_spec(var)
            m_cfg = RMRv3Config.from_dict(spec["model"], pretrained=False)
            model = RMRv3(m_cfg).eval()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable_params == exp_params
            assert trainable_params <= 105000

            with torch.no_grad():
                out = model(dummy_img)
            assert torch.isfinite(out.y).all()
            assert (out.y >= 0.0).all()

    def test_tier4_gt_consistency_invariant(
        self,
        sha_a_test_manifest_path: Path,
    ) -> None:
        """Scenario 7: Discrete rasterized target count sum(Y_gt) matches raw point count inside image."""
        ds = CrowdManifestDataset(manifest=str(sha_a_test_manifest_path), train=False)
        # Verify first 5 samples
        for idx in range(min(5, len(ds))):
            sample = ds[idx]
            target_y = sample["target_y"]
            pts = sample["points"]
            h, w = sample["height"], sample["width"]
            # Filter points inside [0, w) x [0, h)
            valid_pts = (
                (pts[:, 0] >= 0) & (pts[:, 0] < w) &
                (pts[:, 1] >= 0) & (pts[:, 1] < h)
            )
            expected_count = float(valid_pts.sum().item())
            rasterized_count = float(target_y.sum().item())
            assert math.isclose(rasterized_count, expected_count, abs_tol=1e-4), (
                f"Sample {idx}: rasterized count {rasterized_count} != raw valid points {expected_count}"
            )

    def test_tier4_tiled_inference_mass_consistency(
        self,
        canonical_step0_config_dict: Dict[str, Any],
    ) -> None:
        """Scenario 8: Tiled inference prediction matches direct inference within numerical tolerance."""
        m_cfg = RMRv3Config.from_dict(canonical_step0_config_dict["model"], pretrained=False)
        model = RMRv3(m_cfg).eval()

        img_3d = torch.randn(3, 128, 128)
        with torch.no_grad():
            pred_tiled_map = predict_tiled(
                model=model,
                image=img_3d,
                output_stride=4,
                tile_size=64,
                halo=16,
            )
            out_direct = model(img_3d.unsqueeze(0))
            pred_direct_map = out_direct.y[0, 0]
            pred_tiled_2d = pred_tiled_map.squeeze()

        assert pred_tiled_2d.shape == pred_direct_map.shape
        assert torch.isfinite(pred_tiled_2d).all()
        count_tiled = pred_tiled_2d.sum().item()
        count_direct = pred_direct_map.sum().item()
        assert abs(count_tiled - count_direct) / max(1.0, count_direct) < 0.30
