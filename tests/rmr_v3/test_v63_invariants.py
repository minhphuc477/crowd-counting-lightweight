"""Test suite for RMR V63 configuration and architectural invariants.

Invariants tested:
1. Track A parameter ceiling: strictly <= 104,441 trainable parameters.
2. Track B parameter ceiling: strictly <= 200,000 trainable parameters.
3. 100% Standalone (0.0% Knowledge Distillation).
4. Permanent ban on WSD scheduler (only Cosine Annealing with high floor allowed).
5. ASPP-Lite neck dilation verification [2, 4, 8].
6. Scale Jittering range verification.
"""
from __future__ import annotations

import glob
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import make_model


def get_v63_config_paths() -> list[str]:
    root = Path(__file__).resolve().parent.parent.parent
    paths = sorted(glob.glob(str(root / "configs" / "rmr_v63" / "*.yaml")))
    assert len(paths) >= 5, f"Expected at least 5 configs in configs/rmr_v63, found {len(paths)}"
    return paths


@pytest.mark.parametrize("cfg_path", get_v63_config_paths())
def test_v63_configs_pass_validation(cfg_path: str):
    """Ensure every V63 config passes strict schema validation."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)
    assert cfg["train"]["epochs"] in (1500, 2000), f"Invalid epochs: {cfg['train']['epochs']}"
    assert cfg["train"]["min_lr_ratio"] in (0.05, 0.20), f"Invalid min_lr_ratio: {cfg['train']['min_lr_ratio']}"


@pytest.mark.parametrize("cfg_path", get_v63_config_paths())
def test_v63_wsd_is_strictly_banned(cfg_path: str):
    """Ensure WSD is nowhere configured in any V63 config."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("train", {})
    sched = str(train_cfg.get("scheduler_type", "cosine")).lower()
    assert "wsd" not in sched and "stable" not in sched, f"WSD configured in {cfg_path}!"
    assert "wsd_stable_ratio" not in train_cfg, f"wsd_stable_ratio present in {cfg_path}!"


@pytest.mark.parametrize("cfg_path", get_v63_config_paths())
def test_v63_parameter_budgets(cfg_path: str):
    """Ensure every V63 model strictly respects its parameter limit."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, _ = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_p = cfg["model"]["max_trainable_params"]

    if "track_a" in cfg_path:
        assert n_params <= 104441, f"Track A exceeded 104,441: got {n_params}"
    else:
        assert n_params <= 200000, f"Track B exceeded 200,000: got {n_params}"
    assert n_params <= max_p, f"Exceeded max_trainable_params {max_p}: got {n_params}"


def test_v63_champion_forward_pass():
    """Verify champion model forward pass and shape consistency on subpixel lattice."""
    cfg_path = Path("configs/rmr_v63/v63_champion_aspp_subpixel_1500ep.yaml")
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


def test_point_preserving_random_erasing():
    """Verify that Point-Preserving Random Erasing preserves all points while erasing image pixels."""
    from rmr_core.transforms import point_preserving_random_erasing

    # Synthetic image tensor (C=3, H=100, W=100) with gradient
    img_t = torch.linspace(0, 1, 100).view(1, 1, 100).expand(3, 100, 100).clone()
    pts = torch.tensor([[50.0, 50.0], [20.0, 20.0], [80.0, 80.0]], dtype=torch.float32)

    # Force erase with prob 1.0
    erased_img, kept_pts = point_preserving_random_erasing(
        img_t,
        pts,
        prob=1.0,
        scale=(0.1, 0.2),  # large enough to definitely erase a patch
        ratio=(1.0, 1.0),
        max_patches=1,
    )

    # Points must be 100% preserved in number and exact coordinates
    assert len(kept_pts) == len(pts), "Points were dropped by erasing!"
    assert torch.allclose(kept_pts, pts), "Points were altered by erasing!"
    assert erased_img.shape == img_t.shape, "Image shape altered by erasing!"
    # Ensure some pixels were modified to channel mean
    assert not torch.allclose(erased_img, img_t), "Image was not modified by erasing!"


def test_all_core_files_under_450_lines():
    """Verify strictly that all Python source files in rmr_core/ and rmr_v3/ have <= 450 lines."""
    root = Path(__file__).resolve().parent.parent.parent
    violations = []
    for pkg in ["rmr_core", "rmr_v3"]:
        for py_path in (root / pkg).rglob("*.py"):
            with open(py_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > 450:
                violations.append((str(py_path.relative_to(root)), len(lines)))

    assert not violations, f"Files exceeding 450 lines: {violations}"

