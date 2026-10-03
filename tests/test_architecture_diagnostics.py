"""Unit and mathematical invariant tests for scripts/diagnose_architecture_and_errors.py."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch

from scripts.diagnose_architecture_and_errors import (
    audit_aspp_sampling_geometry,
    audit_dataset_collisions,
)


def test_aspp_sampling_geometry_properties():
    """Verify receptive field and sparsity calculations for ASPP-Lite."""
    res = audit_aspp_sampling_geometry(dilations=(1, 3, 6), kernel_size=3, stride=4)
    assert "dilation_1" in res
    assert "dilation_3" in res
    assert "dilation_6" in res

    # Dilation 1
    d1 = res["dilation_1"]
    assert d1["rf_in_feature_grid"] == 3
    assert d1["rf_in_image_pixels"] == 9
    assert d1["sampling_step_pixels"] == 4
    assert d1["convex_hull_area_px"] == 81
    assert d1["gridding_hole_risk"] == "LOW"

    # Dilation 6
    d6 = res["dilation_6"]
    assert d6["rf_in_feature_grid"] == 13
    assert d6["rf_in_image_pixels"] == 49
    assert d6["sampling_step_pixels"] == 24
    assert d6["convex_hull_area_px"] == 2401
    assert d6["gridding_hole_risk"] == "HIGH"
    assert d6["sampling_sparsity_pct"] > 99.0


def test_dataset_collision_audit_synthetic():
    """Verify collision and Rayleigh limit calculation on synthetic ground truth."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        # Image 1: 2 heads very close (distance 3 px < Rayleigh 5.04 px)
        # falling in the same stride-4 cell (0, 0)
        item1 = {
            "image": "img1.jpg",
            "points": [[1.0, 1.0], [3.0, 3.0], [50.0, 50.0]],
            "id": "1",
        }
        # Image 2: 1 head
        item2 = {
            "image": "img2.jpg",
            "points": [[100.0, 100.0]],
            "id": "2",
        }
        f.write(json.dumps(item1) + "\n")
        f.write(json.dumps(item2) + "\n")
        temp_path = f.name

    try:
        res = audit_dataset_collisions(temp_path, strides=(2, 4, 8), rayleigh_cutoff_px=5.04)
        summary = res["dataset_summary"]
        assert summary["total_images"] == 2
        assert summary["total_heads"] == 4

        rayleigh = res["rayleigh_resolution_audit"]
        # The pair (1,1) and (3,3) has distance sqrt(8) approx 2.82 px < 5.04 px
        assert rayleigh["rayleigh_violating_heads"] >= 1

        stride4 = res["stride_collision_audit"]["stride_4"]
        assert stride4["occupied_cells"] == 3  # (0,0), (12,12), (25,25)
        assert stride4["double_head_cells_pct"] > 0.0
        assert stride4["max_heads_in_single_cell"] == 2
    finally:
        Path(temp_path).unlink(missing_ok=True)
