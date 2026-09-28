from __future__ import annotations

from pathlib import Path
import pytest
import torch

from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.engine import make_model, make_loss_cfg
from rmr_v3.losses import compute_rmr_v3_losses


MATRIX_DIR = Path("configs/rmr_matrix")
CONFIG_FILES = sorted(list(MATRIX_DIR.glob("*.yaml")))


def test_matrix_directory_completeness():
    """Verify that all 15 matrix configurations exist."""
    assert len(CONFIG_FILES) == 15, f"Expected 15 matrix configs, found {len(CONFIG_FILES)}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.name)
def test_matrix_config_invariants(config_path: Path):
    """Verify schema, parameter ceiling, official split, zero-KD, and execution integrity."""
    cfg = load_config(str(config_path))
    validate_v3_config(cfg)
    
    # 1. Dataset Split Invariants: Must be 100% official ShanghaiTech Part A
    train_m = cfg["data"]["train_manifest"]
    val_m = cfg["data"]["val_manifest"]
    assert "train_all" in train_m, f"Forbidden ad-hoc train split in {config_path.name}: {train_m}"
    assert "test" in val_m, f"Forbidden non-standard val split in {config_path.name}: {val_m}"
    
    # 2. Strict Zero KD Invariant
    assert cfg.get("train", {}).get("teacher_ckpt") is None, f"KD detected in {config_path.name}"
    
    # 3. Parameter Ceiling Invariant: Strictly <= 105,000
    model, _ = make_model(cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params <= 105000, f"Exceeded parameter ceiling ({trainable_params}) in {config_path.name}"
    
    # 4. Dummy forward and loss execution
    loss_cfg = make_loss_cfg(cfg)
    x = torch.randn(1, 3, 128, 128)
    gt_map = torch.zeros(1, 1, 32, 32)
    gt_map[0, 0, 10, 10] = 1.0
    
    with torch.no_grad():
        out = model(x)
        losses = compute_rmr_v3_losses(out, gt_map, cfg=loss_cfg)
    
    assert not torch.isnan(losses["total"]) and not torch.isinf(losses["total"])
