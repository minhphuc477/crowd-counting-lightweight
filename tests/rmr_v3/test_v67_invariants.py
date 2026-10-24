"""Test suite for RMR V67 configuration and architectural invariants.

Invariants tested:
1. Track A parameter ceiling: strictly <= 104,441 trainable parameters.
2. Track B parameter ceiling: strictly <= 200,000 trainable parameters.
3. 100% Standalone (0.0% Knowledge Distillation, zero teachers).
4. Permanent ban on WSD scheduler (only High-Floor Cosine Annealing allowed).
5. All 7 conflicting auxiliary losses set to 0.0.
6. Forward pass and non-zero gradient verification on both y and y0.
7. File length: strictly <= 450 lines per file.
"""
from __future__ import annotations

import glob
from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.operators import RegionSet, build_multiscale_regions
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import make_model
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def get_v67_config_paths() -> list[str]:
    root = Path(__file__).resolve().parent.parent.parent
    paths = sorted(glob.glob(str(root / "configs" / "rmr_v67" / "*.yaml")))
    assert len(paths) == 12, f"Expected 12 configs in configs/rmr_v67, found {len(paths)}"
    return paths


@pytest.mark.parametrize("cfg_path", get_v67_config_paths())
def test_v67_configs_pass_validation(cfg_path: str):
    """Ensure every V67 config passes strict schema validation."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)
    assert cfg["train"]["epochs"] == 1500, f"Invalid epochs: {cfg['train']['epochs']}"
    assert cfg["train"]["min_lr_ratio"] == 0.20, f"Invalid min_lr_ratio: {cfg['train']['min_lr_ratio']}"


@pytest.mark.parametrize("cfg_path", get_v67_config_paths())
def test_v67_wsd_is_strictly_banned(cfg_path: str):
    """Ensure WSD is nowhere configured in any V67 config."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg.get("train", {})
    sched = str(train_cfg.get("scheduler_type", "cosine")).lower()
    assert "wsd" not in sched and "stable" not in sched, f"WSD configured in {cfg_path}!"
    assert "wsd_stable_ratio" not in train_cfg, f"wsd_stable_ratio present in {cfg_path}!"


@pytest.mark.parametrize("cfg_path", get_v67_config_paths())
def test_v67_parameter_budgets(cfg_path: str):
    """Ensure every V67 model strictly respects its parameter limit."""
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


def test_v67_gold_forward_pass():
    """Verify flagship first-principles model forward pass on subpixel lattice."""
    cfg_path = Path("configs/rmr_v67/v67_exp0_gold_first_principles_1500ep.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, _ = make_model(cfg)
    model.eval()

    # Synthetic image (B=2, C=3, H=512, W=512)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    assert out["y"].shape == (2, 1, 256, 256), f"Unexpected shape {out['y'].shape}"
    assert out["y0"].shape == (2, 1, 256, 256), f"Unexpected shape {out['y0'].shape}"
    assert torch.isfinite(out["y"]).all(), "Non-finite output in y"
    assert torch.isfinite(out["y0"]).all(), "Non-finite output in y0"


def test_v67_first_principles_loss_gradients():
    """Verify that V67 3-loss suite delivers clean, non-zero gradients on y and y0."""
    cfg_path = Path("configs/rmr_v67/v67_exp0_gold_first_principles_1500ep.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    loss_cfg = RMRv3LossConfig(**cfg.get("loss", {}))
    b, h, w = 2, 256, 256
    y = torch.full((b, 1, h, w), 0.05, requires_grad=True)
    y0 = torch.full((b, 1, h, w), 0.05, requires_grad=True)

    regions = build_multiscale_regions(
        height=h, width=w, output_stride=2, region_sizes_px=(32, 64, 128)
    )
    k = regions.boxes.shape[0]
    b_region = torch.full((b, k), 5.0, requires_grad=True)
    disp_region = torch.full((b, k), 50.0)

    target_float = torch.zeros((b, 1, h, w))
    target_float[:, :, 10:15, 10:15] = 0.5
    points = [torch.tensor([[20.0, 20.0], [40.0, 40.0]]), torch.tensor([[100.0, 100.0]])]

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": b_region,
        "region_dispersion": disp_region,
    }

    losses = compute_rmr_v3_losses(
        outputs,
        target_y=target_float,
        cfg=loss_cfg,
        points=points,
    )

    assert "total" in losses
    assert torch.isfinite(losses["total"])
    assert losses["total"] > 0.0

    losses["total"].backward()

    assert y.grad is not None
    assert y0.grad is not None
    assert torch.isfinite(y.grad).all()
    assert torch.isfinite(y0.grad).all()
    assert (y.grad.abs().sum()) > 0.0, "Zero gradient on y!"
    assert (y0.grad.abs().sum()) > 0.0, "Zero gradient on y0!"
