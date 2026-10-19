from __future__ import annotations

from pathlib import Path
import yaml
import torch
import pytest

from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


V57_CONFIGS = [
    "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml",
    "configs/rmr_v57/v57_track_a_ablation_1step_rn.yaml",
    "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml",
    "configs/rmr_v57/v57_track_b_ablation_1step_rn.yaml",
]


@pytest.mark.parametrize("cfg_path", V57_CONFIGS)
def test_v57_config_loading_and_parameter_budgets(cfg_path: str):
    """Verify that all V57 configs load validly and strictly satisfy parameter budgets."""
    path = Path(cfg_path)
    assert path.is_file(), f"Missing config file: {cfg_path}"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

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


@pytest.mark.parametrize("cfg_path", V57_CONFIGS)
def test_v57_standalone_no_distillation(cfg_path: str):
    """Verify that all V57 configs enforce 100% standalone training (0.0% KD)."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_keys = str(list(data.get("model", {}).keys()) + list(data.get("loss", {}).keys())).lower()
    assert "distill" not in raw_keys, f"Distillation key found in {cfg_path}"
    assert "teacher" not in raw_keys, f"Teacher key found in {cfg_path}"
    assert "kd" not in raw_keys, f"KD key found in {cfg_path}"


def test_v57_multistep_rn_and_faithful_beta_nb_config():
    """Verify 2-step unrolling and faithful beta-nb parameters in V57 proposed configs."""
    # Track A Proposed
    with open("configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml", "r", encoding="utf-8") as f:
        data_a = yaml.safe_load(f)
    m_cfg_a = RMRv3Config.from_dict(data_a.get("model", {}))
    l_cfg_a = RMRv3LossConfig.from_dict(data_a.get("loss", {}))

    assert m_cfg_a.rn_iterations == 2
    assert m_cfg_a.rn_trust_alpha == 0.12
    assert m_cfg_a.rn_trust_alpha_up == 0.35
    assert l_cfg_a.faithful_regional_nb is True
    assert l_cfg_a.regional_nb_beta == 0.5

    # Track B Proposed
    with open("configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml", "r", encoding="utf-8") as f:
        data_b = yaml.safe_load(f)
    m_cfg_b = RMRv3Config.from_dict(data_b.get("model", {}))
    l_cfg_b = RMRv3LossConfig.from_dict(data_b.get("loss", {}))

    assert m_cfg_b.regional_feature_stats == "mean_std"
    assert m_cfg_b.rn_iterations == 2
    assert l_cfg_b.faithful_regional_nb is True
    assert l_cfg_b.regional_nb_beta == 0.5


def test_v57_track_a_multistep_forward_backward():
    """Verify forward and backward passes on Track A Proposed with 2-step RN and faithful Beta-NB."""
    cfg_path = "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml"
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

    # 2 RN iterations -> iterates has [y0, y1, y2]
    assert len(out.iterates) == 3
    assert out.y.shape == (2, 1, 32, 32)
    assert out.rn_multiplier is not None

    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    target_y[0, 0, 5, 5] = 1.0
    target_y[1, 0, 10, 10] = 3.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0], [42.0, 42.0], [44.0, 44.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)

    assert losses["count"].item() > 0.0
    assert losses["flat_dm16"].item() > 0.0
    assert losses["region_nb"].item() > 0.0
    assert losses["rn_transport"].item() >= 0.0

    losses["total"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients were backpropagated through Track A multi-step model"


def test_v57_track_b_texture_multistep_forward_backward():
    """Verify forward and backward passes on Track B Proposed with mean_std and 2-step RN."""
    cfg_path = "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_dict = dict(data.get("model", {}))
    model_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(model_dict)
    l_cfg = RMRv3LossConfig.from_dict(data.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Verify parameter count with mean_std
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 140000 <= total_params <= 200000, f"Expected 140k-200k params, got {total_params}"

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert len(out.iterates) == 3
    assert out.y.shape == (2, 1, 64, 64)
    assert out.rn_multiplier is not None

    target_y = torch.zeros(2, 1, 64, 64, dtype=torch.float32)
    target_y[0, 0, 10, 10] = 1.0
    target_y[1, 0, 20, 20] = 3.0
    pts = [torch.tensor([[20.0, 20.0]]), torch.tensor([[40.0, 40.0], [42.0, 42.0], [44.0, 44.0]])]

    losses = compute_rmr_v3_losses(out, target_y, l_cfg, points=pts)

    assert losses["count"].item() > 0.0
    assert losses["flat_dm16"].item() > 0.0
    assert losses["region_nb"].item() > 0.0
    assert losses["rn_transport"].item() >= 0.0

    losses["total"].backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradients were backpropagated through Track B texture model"
