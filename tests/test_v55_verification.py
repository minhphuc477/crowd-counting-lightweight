from __future__ import annotations

from pathlib import Path
import yaml
import torch
import pytest

from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


V55_CONFIGS = [
    "configs/rmr_v55/v55_track_a_purified_rn.yaml",
    "configs/rmr_v55/v55_track_a_no_transport_loss.yaml",
    "configs/rmr_v55/v55_track_a_control_bloat.yaml",
    "configs/rmr_v55/v55_track_b_purified_rn.yaml",
    "configs/rmr_v55/v55_track_b_control_bloat.yaml",
]


@pytest.mark.parametrize("cfg_path", V55_CONFIGS)
def test_v55_config_loading_and_parameter_budgets(cfg_path: str):
    """Verify that all V55 configs load validly and strictly satisfy parameter budgets."""
    path = Path(cfg_path)
    assert path.is_file(), f"Missing config file: {cfg_path}"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    m_cfg = RMRv3Config.from_dict(data.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if "track_a" in cfg_path:
        assert total_params <= 104441, f"Track A budget violated: {total_params} > 104,441 in {cfg_path}"
    elif "track_b" in cfg_path:
        assert total_params <= 200000, f"Track B budget violated: {total_params} > 200,000 in {cfg_path}"


@pytest.mark.parametrize("cfg_path", V55_CONFIGS)
def test_v55_standalone_no_distillation(cfg_path: str):
    """Verify that all V55 configs enforce 100% standalone training (0.0% KD)."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_keys = str(list(data.get("model", {}).keys()) + list(data.get("loss", {}).keys())).lower()
    assert "distill" not in raw_keys, f"Distillation key found in {cfg_path}"
    assert "teacher" not in raw_keys, f"Teacher key found in {cfg_path}"
    assert "kd" not in raw_keys, f"KD key found in {cfg_path}"


def test_v55_track_a_purified_forward_backward():
    """Verify forward pass, purified 4-loss computation, and backward pass on Track A."""
    cfg_path = "configs/rmr_v55/v55_track_a_purified_rn.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    m_cfg = RMRv3Config.from_dict(data.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "rn_multiplier" in out
    assert out.rn_multiplier is not None

    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    target_y[0, 0, 5, 5] = 1.0
    target_y[1, 0, 10, 10] = 2.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0], [42.0, 42.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)

    # Verify only 4 core losses are active
    assert losses["count"].item() > 0.0
    assert losses["flat_dm16"].item() > 0.0
    assert losses["region_nb"].item() > 0.0
    assert losses["rn_transport"].item() >= 0.0

    # Verify all auxiliary crutches are exactly 0
    assert losses["cell"].item() == pytest.approx(0.0)
    assert losses["hurdle_bce"].item() == pytest.approx(0.0)
    assert losses["trunc_nb"].item() == pytest.approx(0.0)
    assert losses["scale_align"].item() == pytest.approx(0.0)
    assert losses["curvature"].item() == pytest.approx(0.0)
    assert losses["hard_bg"].item() == pytest.approx(0.0)

    losses["total"].backward()

    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))
            has_grad = True
    assert has_grad


def test_v55_track_b_purified_forward_backward():
    """Verify forward pass, subpixel stride 2, and backward pass on Track B."""
    cfg_path = "configs/rmr_v55/v55_track_b_purified_rn.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    m_cfg = RMRv3Config.from_dict(data.get("model", {}))
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert out.y.shape[-2:] == (64, 64)
    assert out.rn_multiplier is not None

    target_y = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    target_y[0, 0, 10, 10] = 1.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)
    assert "rn_transport" in losses
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad))
            has_grad = True
    assert has_grad
