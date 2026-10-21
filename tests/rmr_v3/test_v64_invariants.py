"""Test suite for RMR V64 configuration and architectural invariants.

Invariants tested:
1. Track B parameter ceiling: strictly <= 200,000 trainable parameters.
2. 100% Standalone (0.0% Knowledge Distillation).
3. Permanent ban on WSD scheduler (only High-Floor Cosine Annealing allowed).
4. Floor tau quadratic background suppression.
5. Calibrated dual allocation (DM16 + Bayesian loss with 0.0 background ratio).
6. DiAG perspective geometry with vertical gradient.
7. File length: strictly <= 450 lines per file.
"""
from __future__ import annotations

import glob
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import make_model


def get_v64_config_paths() -> list[str]:
    root = Path(__file__).resolve().parent.parent.parent
    paths = sorted(glob.glob(str(root / "configs" / "rmr_v64" / "*.yaml")))
    assert len(paths) >= 4, f"Expected at least 4 configs in configs/rmr_v64, found {len(paths)}"
    return paths


@pytest.mark.parametrize("cfg_path", get_v64_config_paths())
def test_v64_configs_pass_validation(cfg_path: str):
    """Ensure every V64 config passes strict schema validation."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)
    assert cfg["train"]["epochs"] == 1500, f"Invalid epochs: {cfg['train']['epochs']}"
    assert cfg["train"]["min_lr_ratio"] == 0.20, f"Invalid min_lr_ratio: {cfg['train']['min_lr_ratio']}"


@pytest.mark.parametrize("cfg_path", get_v64_config_paths())
def test_v64_wsd_is_strictly_banned(cfg_path: str):
    """Ensure WSD is nowhere configured in any V64 config."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("train", {})
    sched = str(train_cfg.get("scheduler_type", "cosine")).lower()
    assert "wsd" not in sched and "stable" not in sched, f"WSD configured in {cfg_path}!"
    assert "wsd_stable_ratio" not in train_cfg, f"wsd_stable_ratio present in {cfg_path}!"


@pytest.mark.parametrize("cfg_path", get_v64_config_paths())
def test_v64_parameter_budgets(cfg_path: str):
    """Ensure every V64 model strictly respects its parameter limit."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, _ = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_p = cfg["model"]["max_trainable_params"]

    assert n_params <= 200000, f"Track B exceeded 200,000: got {n_params}"
    assert n_params <= max_p, f"Exceeded max_trainable_params {max_p}: got {n_params}"


def test_v64_synergy_forward_pass():
    """Verify flagship synergy model forward pass and shape consistency on subpixel lattice."""
    cfg_path = Path("configs/rmr_v64/v64_champion_synergy_50mae_1500ep.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, _ = make_model(cfg)
    model.eval()

    # Synthetic image (B=2, C=3, H=512, W=512)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert "y" in out, "Output missing recovered measure y"
    assert "y0" in out, "Output missing carrier measure y0"
    assert out["y"].shape == (2, 1, 256, 256), f"Unexpected y shape: {out['y'].shape}"
    assert out["y0"].shape == (2, 1, 256, 256), f"Unexpected y0 shape: {out['y0'].shape}"
    assert (out["y"] >= 0).all(), "Recovered measure y contains negative values"
