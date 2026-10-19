from __future__ import annotations

from pathlib import Path
import yaml
import torch
import pytest

from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


V56_CONFIGS = [
    "configs/rmr_v56/v56_track_a_proposed_asym_rn.yaml",
    "configs/rmr_v56/v56_track_a_ablation_sym_rn.yaml",
    "configs/rmr_v56/v56_track_b_proposed_deep_asym_rn.yaml",
    "configs/rmr_v56/v56_track_b_ablation_sym_rn.yaml",
]


@pytest.mark.parametrize("cfg_path", V56_CONFIGS)
def test_v56_config_loading_and_parameter_budgets(cfg_path: str):
    """Verify that all V56 configs load validly and strictly satisfy parameter budgets."""
    path = Path(cfg_path)
    assert path.is_file(), f"Missing config file: {cfg_path}"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Disable pretrained download during test
    model_dict = dict(data.get("model", {}))
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)
    _ = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if "track_a" in cfg_path:
        assert total_params <= 104441, f"Track A budget violated: {total_params} > 104,441 in {cfg_path}"
    elif "track_b" in cfg_path:
        assert total_params <= 200000, f"Track B budget violated: {total_params} > 200,000 in {cfg_path}"


@pytest.mark.parametrize("cfg_path", V56_CONFIGS)
def test_v56_standalone_no_distillation(cfg_path: str):
    """Verify that all V56 configs enforce 100% standalone training (0.0% KD)."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_keys = str(list(data.get("model", {}).keys()) + list(data.get("loss", {}).keys())).lower()
    assert "distill" not in raw_keys, f"Distillation key found in {cfg_path}"
    assert "teacher" not in raw_keys, f"Teacher key found in {cfg_path}"
    assert "kd" not in raw_keys, f"KD key found in {cfg_path}"


def test_v56_asymmetric_trust_configuration():
    """Verify asymmetric trust configuration in proposed vs ablation configs."""
    # Proposed Track A
    with open("configs/rmr_v56/v56_track_a_proposed_asym_rn.yaml", "r", encoding="utf-8") as f:
        data_prop = yaml.safe_load(f)
    m_cfg_prop = RMRv3Config.from_dict(data_prop.get("model", {}))
    assert m_cfg_prop.rn_trust_alpha == 0.12
    assert m_cfg_prop.rn_trust_alpha_up == 0.35

    # Ablation Track A
    with open("configs/rmr_v56/v56_track_a_ablation_sym_rn.yaml", "r", encoding="utf-8") as f:
        data_abl = yaml.safe_load(f)
    m_cfg_abl = RMRv3Config.from_dict(data_abl.get("model", {}))
    assert m_cfg_abl.rn_trust_alpha == 0.12
    assert m_cfg_abl.rn_trust_alpha_up is None


def test_v56_track_b_deep_capacity():
    """Verify Track B expanded capacity: 3 head blocks, hidden 96, and >130k parameters."""
    with open("configs/rmr_v56/v56_track_b_proposed_deep_asym_rn.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_dict = dict(data.get("model", {}))
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)
    assert m_cfg.head_blocks == 3
    assert m_cfg.region_head_hidden == 96

    model = RMRv3(m_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 130000 <= total_params <= 200000, f"Expected 130k-200k params, got {total_params}"


def test_v56_track_a_proposed_forward_backward():
    """Verify forward pass, loss computation, and backward pass on Track A Proposed."""
    cfg_path = "configs/rmr_v56/v56_track_a_proposed_asym_rn.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_dict = dict(data.get("model", {}))
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert out.rn_multiplier is not None

    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    target_y[0, 0, 5, 5] = 1.0
    target_y[1, 0, 10, 10] = 2.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0], [42.0, 42.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)

    assert losses["count"].item() > 0.0
    assert losses["flat_dm16"].item() > 0.0
    assert losses["region_nb"].item() > 0.0
    assert losses["rn_transport"].item() >= 0.0

    losses["total"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients were backpropagated through Track A model"


def test_v56_track_b_proposed_forward_backward():
    """Verify forward pass, loss computation, and backward pass on Track B Proposed (Stride 2)."""
    cfg_path = "configs/rmr_v56/v56_track_b_proposed_deep_asym_rn.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_dict = dict(data.get("model", {}))
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    # Stride 2 output for 128x128 input is 64x64
    assert out.y.shape == (2, 1, 64, 64)

    target_y = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    target_y[0, 0, 10, 10] = 1.0
    target_y[1, 0, 20, 20] = 2.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0], [42.0, 42.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)

    assert losses["count"].item() > 0.0
    assert losses["flat_dm16"].item() > 0.0
    assert losses["region_nb"].item() > 0.0
    assert losses["rn_transport"].item() >= 0.0

    losses["total"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients were backpropagated through Track B model"
