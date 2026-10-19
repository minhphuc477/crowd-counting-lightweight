"""Comprehensive test suite for V57 Experiment Configurations and Architectural Invariants.

Verifies:
1. YAML schema conformance under validate_v3_config across all active research configs.
2. Strict adherence to parameter budgets (Track A <= 104,441, Track B <= 200,000).
3. Zero ad-hoc split policy (data/sha_a_train_all.jsonl, data/sha_a_test.jsonl).
4. End-to-end forward pass, loss orchestration, and backward gradient flow.
5. 2-Step Radon-Nikodym contraction mapping dynamics.
"""
from __future__ import annotations

import glob
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.model.architecture import RMRv3
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


V57_CONFIGS = [
    "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml",
    "configs/rmr_v57/v57_track_a_ablation_1step_rn.yaml",
    "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml",
    "configs/rmr_v57/v57_track_b_ablation_1step_rn.yaml",
]

V53_TO_V57_CONFIGS = sorted(
    glob.glob("configs/rmr_v5*/**/*.yaml", recursive=True)
    + glob.glob("configs/rmr_v5*/*.yaml")
)
V53_TO_V57_CONFIGS = sorted(list(set(V53_TO_V57_CONFIGS)))


@pytest.mark.parametrize("cfg_path", V57_CONFIGS)
def test_v57_yaml_validity(cfg_path: str):
    """Verify all V57 YAML configs conform strictly to schema and pass validation."""
    p = Path(cfg_path)
    assert p.is_file(), f"Config file missing: {cfg_path}"
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)


@pytest.mark.parametrize("cfg_path", V57_CONFIGS)
def test_v57_manifest_invariants(cfg_path: str):
    """Verify official ShanghaiTech Part A partitions and zero ad-hoc splits."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    tm = d.get("train_manifest", "")
    vm = d.get("val_manifest", "")
    assert tm == "data/sha_a_train_all.jsonl", f"Invalid train manifest: {tm}"
    assert vm == "data/sha_a_test.jsonl", f"Invalid val manifest: {vm}"
    assert Path(tm).is_file(), f"Train manifest missing on disk: {tm}"
    assert Path(vm).is_file(), f"Val manifest missing on disk: {vm}"


@pytest.mark.parametrize("cfg_path", V57_CONFIGS)
def test_v57_model_parameter_budgets(cfg_path: str):
    """Verify strict parameter ceilings: Track A <= 104,441 and Track B <= 200,000."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_dict = dict(cfg.get("model", {}))
    m_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(m_dict)
    model = RMRv3(m_cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if m_cfg.output_stride == 4:
        assert trainable_params <= 104441, (
            f"Track A parameter ceiling violation in {cfg_path}: "
            f"{trainable_params} > 104,441"
        )
    elif m_cfg.output_stride == 2:
        assert trainable_params <= 200000, (
            f"Track B parameter ceiling violation in {cfg_path}: "
            f"{trainable_params} > 200,000"
        )


@pytest.mark.parametrize("cfg_path", [
    "configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml",
    "configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml",
])
def test_v57_forward_backward_loss_flow(cfg_path: str):
    """Verify forward pass, loss orchestration, and finite backward gradients."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_dict = dict(cfg.get("model", {}))
    m_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(m_dict)
    l_cfg = RMRv3LossConfig.from_dict(cfg.get("loss", {}))

    model = RMRv3(m_cfg)
    model.train()

    # Synthetic batch of 2 images of size 256x256
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    h_out, w_out = out.y.shape[-2:]
    expected_stride = m_cfg.output_stride
    assert h_out == 256 // expected_stride
    assert w_out == 256 // expected_stride

    target_y = torch.zeros(2, 1, h_out, w_out)
    target_y[0, 0, 10, 10] = 1.0
    target_y[0, 0, 20, 20] = 1.0
    target_y[1, 0, 15, 15] = 1.0
    points = [torch.tensor([[40.0, 40.0], [80.0, 80.0]]), torch.tensor([[60.0, 60.0]])]

    losses = compute_rmr_v3_losses(outputs=out, target_y=target_y, cfg=l_cfg, points=points)
    total_loss = losses["total"]
    assert torch.isfinite(total_loss), f"Non-finite loss in {cfg_path}: {total_loss}"

    total_loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), "NaN or Inf detected in gradients"
    assert has_grad, "No gradients were accumulated"


def test_v57_multistep_contraction_dynamics():
    """Verify that rn_iterations=2 produces two sequential contraction updates."""
    with open("configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m_dict = dict(cfg.get("model", {}))
    m_dict["pretrained"] = False
    m_cfg = RMRv3Config.from_dict(m_dict)
    assert m_cfg.rn_iterations == 2
    assert m_cfg.rn_trust_alpha == 0.12
    assert m_cfg.rn_trust_alpha_up == 0.35

    model = RMRv3(m_cfg)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    # iterates contains [y0, y_step1, y_step2]
    assert len(out.iterates) == 3
    assert out.rn_multiplier is not None
    assert (out.rn_multiplier >= 0.0).all()
    assert (out.y >= 0.0).all()


@pytest.mark.parametrize("cfg_path", V53_TO_V57_CONFIGS)
def test_all_v53_to_v57_configs_validate(cfg_path: str):
    """Verify that all 51 configurations in rmr_v53-v57 pass validation."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)
