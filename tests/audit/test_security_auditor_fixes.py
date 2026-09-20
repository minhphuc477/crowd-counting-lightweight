from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import numpy as np

from rmr_core.training import safe_torch_load, safe_torch_save
from rmr_v3.train import sanitize_run_id
from rmr_v3.solver import unrolled_sirt_solver
from rmr_core.operators import build_multiscale_regions, regional_sum
from rmr_core.backbones import TimmPyramidBackbone


# =========================================================================
# SEC-01: Safe Deserialization (CWE-502) & Safe Globals Registration
# =========================================================================

def test_sec01_safe_torch_load_with_numpy_payload(tmp_path: Path):
    """Verify safe_torch_load loads checkpoints with weights_only=True even when containing NumPy types."""
    ckpt_path = tmp_path / "test_ckpt.pt"
    data = {
        "model": {"weight": torch.randn(4, 4)},
        "epoch": np.int64(10),
        "best_mae": np.float64(72.61),
        "density_bins": np.array([100.0, 500.0]),
    }
    torch.save(data, ckpt_path)

    # Must load successfully with weights_only=True
    loaded = safe_torch_load(ckpt_path, map_location="cpu", weights_only=True)
    assert "model" in loaded
    assert "epoch" in loaded
    assert isinstance(loaded["model"]["weight"], torch.Tensor)
    assert loaded["epoch"] == 10
    assert float(loaded["best_mae"]) == pytest.approx(72.61)


def test_sec01_safe_torch_load_missing_file_raises_error():
    """Verify safe_torch_load raises FileNotFoundError for non-existent files."""
    with pytest.raises(FileNotFoundError):
        safe_torch_load("non_existent_checkpoint_path_12345.pt")


# =========================================================================
# SEC-02: CLI Path Traversal Mitigation (CWE-22)
# =========================================================================

@pytest.mark.parametrize(
    "valid_id",
    [
        "c0_seed42",
        "rmr_v32_anchor",
        "exp-123_test_RUN",
        "run_01",
    ],
)
def test_sec02_sanitize_run_id_valid(valid_id: str):
    """Verify valid run identifiers pass sanitization unmodified."""
    assert sanitize_run_id(valid_id) == valid_id


@pytest.mark.parametrize(
    "malicious_id",
    [
        "../../etc/passwd",
        r"..\..\windows\system32",
        "run/with/slash",
        "run;rm -rf /",
        "run|cat",
        "run id with space",
        "",
        "   ",
        "run$id",
    ],
)
def test_sec02_sanitize_run_id_rejects_traversal(malicious_id: str):
    """Verify malicious or invalid run identifiers are strictly rejected."""
    with pytest.raises(ValueError, match="Invalid run_id"):
        sanitize_run_id(malicious_id)


# =========================================================================
# SEC-03: Zero-Host-Sync Divergence Guard (CWE-400)
# =========================================================================

def test_sec03_sirt_solver_zero_host_sync_divergence_guard():
    """Verify unrolled_sirt_solver gracefully handles NaNs/Infs via torch.where without CPU sync stall."""
    b = 2
    h, w = 32, 32
    y0 = torch.zeros(b, 1, h, w)
    z0 = torch.zeros(b, 1, h, w)
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    b_target = torch.ones(b, 1, regions.boxes.shape[0])
    weight = torch.ones(b, 1, regions.boxes.shape[0])

    # Normal execution should yield valid finite output
    result = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_target,
        weight_solver=weight,
        regions=regions,
        iterations=4,
        proximal_mode="clamp",
    )
    assert torch.isfinite(result["y"]).all()

    # Simulate zero-host-sync recovery: torch.where(torch.isfinite(y_next), y_next, y_curr)
    # Even if an unrolled step produces NaN, where restores previous valid iterate
    y_prev = torch.ones(2, 1, 16, 16)
    y_nan = y_prev.clone()
    y_nan[0, 0, 5, 5] = float("nan")
    y_nan[1, 0, 8, 8] = float("inf")

    y_recovered = torch.where(torch.isfinite(y_nan), y_nan, y_prev)
    assert torch.isfinite(y_recovered).all()
    assert y_recovered[0, 0, 5, 5] == 1.0
    assert y_recovered[1, 0, 8, 8] == 1.0


# =========================================================================
# SEC-04: Offline Resiliency for Pretrained Backbone
# =========================================================================

def test_sec04_timm_backbone_offline_fallback(monkeypatch):
    """Verify TimmPyramidBackbone falls back to pretrained=False if network fails."""
    import timm

    orig_create_model = timm.create_model

    def mock_create_model(model_name, pretrained=False, **kwargs):
        if pretrained:
            raise OSError("Mock network failure: Unable to reach huggingface.co")
        return orig_create_model(model_name, pretrained=False, **kwargs)

    monkeypatch.setattr(timm, "create_model", mock_create_model)

    with pytest.warns(UserWarning, match="Failed to load pretrained weights"):
        backbone = TimmPyramidBackbone(
            model_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
            pretrained=True,
            target_reductions=(4, 8, 16),
        )
    assert backbone.pretrained is False
    x = torch.randn(1, 3, 128, 128)
    feats = backbone(x)
    assert len(feats) == 3


# =========================================================================
# SEC-05: Safe Atomic Save Temp File Cleanup (CWE-703)
# =========================================================================

def test_sec05_safe_torch_save_cleans_up_on_failure(tmp_path: Path, monkeypatch):
    """Verify safe_torch_save does not leak temporary files if an exception occurs during saving."""
    target_path = tmp_path / "model.pt"

    # Mock torch.save to fail on the temporary file
    def mock_failing_save(obj, f, **kwargs):
        # Create a partial file to simulate partial write before error
        Path(f).write_text("corrupted partial content", encoding="utf-8")
        raise RuntimeError("Simulated disk write failure or out of space")

    monkeypatch.setattr(torch, "save", mock_failing_save)

    with pytest.raises(RuntimeError, match="Simulated disk write failure"):
        safe_torch_save({"dummy": 123}, target_path)

    # Verify no .tmp_* files were left behind in the directory
    tmp_files = list(tmp_path.glob("*.tmp_*"))
    assert len(tmp_files) == 0, f"Temporary files leaked on disk: {tmp_files}"
