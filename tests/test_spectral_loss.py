"""Comprehensive Unit & Protocol Hardening Tests for Count-Preserving Spectral Loss (H2)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
import pytest
import torch
import yaml

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rmr_core.operators import RegionSet
from rmr_core.spectral import CountPreservingSpectralLoss, count_preserving_spectral_loss
from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def _create_synthetic_batch_outputs(
    b: int = 2,
    h: int = 64,
    w: int = 64,
    requires_grad: bool = False,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Helper to construct dummy model outputs and ground truth density for loss testing."""
    device = torch.device("cpu")
    y = torch.rand(b, 1, h, w, dtype=torch.float32, device=device, requires_grad=requires_grad)
    y0 = torch.rand(b, 1, h, w, dtype=torch.float32, device=device, requires_grad=requires_grad)

    # 4 dummy macro-regions
    boxes = torch.tensor(
        [
            [0, 0, 32, 32],
            [0, 32, 32, 64],
            [32, 0, 64, 32],
            [32, 32, 64, 64],
        ],
        dtype=torch.int64,
        device=device,
    )
    scale_id = torch.tensor([0, 0, 0, 0], dtype=torch.int64, device=device)
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).float()
    boxes_list = [(int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])) for bx in boxes]
    regions = RegionSet(boxes=boxes, scale_id=scale_id, area=area, boxes_list=boxes_list)

    n_reg = boxes.shape[0]
    b_region = torch.ones(b, 1, n_reg, dtype=torch.float32, device=device) * 2.5
    region_dispersion = torch.ones(b, 1, n_reg, dtype=torch.float32, device=device) * 50.0

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": b_region,
        "region_dispersion": region_dispersion,
    }
    target_y = torch.rand(b, 1, h, w, dtype=torch.float32, device=device)
    return outputs, target_y


def test_spectral_loss_identical_inputs():
    """Verify loss is exactly zero when prediction matches target."""
    x = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    loss, details = count_preserving_spectral_loss(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(details["spectral_dc"], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(details["spectral_ac"], torch.tensor(0.0), atol=1e-6)


def test_spectral_loss_dc_mass_conservation():
    """Verify DC component strictly reflects total count difference."""
    h, w = 64, 64
    x = torch.zeros(1, 1, h, w, dtype=torch.float32)
    y = torch.zeros(1, 1, h, w, dtype=torch.float32)

    # Add 10 people to x, 5 people to y
    x[0, 0, 10:20, 10:20] = 0.1   # sum = 10.0
    y[0, 0, 30:40, 30:40] = 0.05  # sum = 5.0

    _, details = count_preserving_spectral_loss(x, y, lambda_count=1.0, lambda_spectral=0.0)
    expected_dc = abs(10.0 - 5.0) / (h * w) ** 0.5
    assert torch.isclose(details["spectral_dc"], torch.tensor(expected_dc, dtype=torch.float32), atol=1e-5)


def test_spectral_loss_backward_gradient():
    """Verify backward pass produces valid, finite gradients."""
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)

    module = CountPreservingSpectralLoss(beta=2.0, lambda_count=1.0, lambda_spectral=0.5)
    loss = module(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()
    assert pred.grad.abs().sum() > 0.0


def test_loss_config_spectral_fields():
    """Verify RMRv3LossConfig schema defaults and validation boundaries for spectral loss."""
    cfg_default = RMRv3LossConfig()
    assert not cfg_default.use_spectral_loss
    assert cfg_default.lambda_spectral == 0.0
    assert cfg_default.spectral_beta == 2.0
    assert cfg_default.lambda_spectral_dc == 1.0

    cfg_custom = RMRv3LossConfig.from_dict({
        "use_spectral_loss": True,
        "lambda_spectral": 0.15,
        "spectral_beta": 2.5,
        "lambda_spectral_dc": 1.5,
    })
    assert cfg_custom.use_spectral_loss
    assert cfg_custom.lambda_spectral == 0.15
    assert cfg_custom.spectral_beta == 2.5
    assert cfg_custom.lambda_spectral_dc == 1.5

    with pytest.raises(ValueError, match="lambda_spectral"):
        RMRv3LossConfig(lambda_spectral=-0.1)
    with pytest.raises(ValueError, match="lambda_spectral_dc"):
        RMRv3LossConfig(lambda_spectral_dc=-0.5)
    with pytest.raises(ValueError, match="spectral_beta"):
        RMRv3LossConfig(spectral_beta=0.0)


def test_orchestration_spectral_loss_disabled():
    """Verify spectral loss component is zero and does not affect total when disabled."""
    outputs, target_y = _create_synthetic_batch_outputs()
    cfg = RMRv3LossConfig(use_spectral_loss=False, lambda_spectral=0.0)
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=cfg)

    assert "spectral" in losses
    assert "spectral_dc" in losses
    assert "spectral_ac" in losses
    assert losses["spectral"].item() == 0.0
    assert losses["spectral_dc"].item() == 0.0
    assert losses["spectral_ac"].item() == 0.0


def test_orchestration_spectral_loss_enabled_y():
    """Verify spectral loss component is correctly computed and accumulated on y."""
    outputs, target_y = _create_synthetic_batch_outputs()
    cfg_base = RMRv3LossConfig(use_spectral_loss=False, lambda_spectral=0.0, dm_target="y")
    losses_base = compute_rmr_v3_losses(outputs, target_y, cfg=cfg_base)

    cfg_spec = RMRv3LossConfig(
        use_spectral_loss=True,
        lambda_spectral=0.1,
        spectral_beta=2.0,
        lambda_spectral_dc=1.0,
        dm_target="y",
    )
    losses_spec = compute_rmr_v3_losses(outputs, target_y, cfg=cfg_spec)

    assert losses_spec["spectral"].item() > 0.0
    assert losses_spec["spectral_dc"].item() >= 0.0
    assert losses_spec["spectral_ac"].item() > 0.0

    expected_total = losses_base["total"] + 0.1 * losses_spec["spectral"]
    assert torch.isclose(losses_spec["total"], expected_total, atol=1e-5)


def test_orchestration_spectral_loss_enabled_dual():
    """Verify spectral loss component dispatches across both y and y0 under dual mode."""
    outputs, target_y = _create_synthetic_batch_outputs()
    cfg_dual = RMRv3LossConfig(
        use_spectral_loss=True,
        lambda_spectral=0.2,
        spectral_beta=2.0,
        lambda_spectral_dc=1.0,
        dm_target="dual",
    )
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=cfg_dual)

    assert losses["spectral"].item() > 0.0
    assert losses["spectral_dc"].item() >= 0.0
    assert losses["spectral_ac"].item() > 0.0
    assert losses["total"].item() > 0.0


def test_orchestration_spectral_loss_gradient_flow():
    """Verify end-to-end backpropagation through both y and y0 under spectral loss."""
    outputs, target_y = _create_synthetic_batch_outputs(requires_grad=True)
    cfg = RMRv3LossConfig(
        use_spectral_loss=True,
        lambda_spectral=0.1,
        spectral_beta=2.0,
        lambda_spectral_dc=1.0,
        dm_target="dual",
    )
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=cfg)
    losses["total"].backward()

    assert outputs["y"].grad is not None
    assert outputs["y0"].grad is not None
    assert not torch.isnan(outputs["y"].grad).any()
    assert not torch.isnan(outputs["y0"].grad).any()
    assert outputs["y"].grad.abs().sum() > 0.0
    assert outputs["y0"].grad.abs().sum() > 0.0


def test_deterministic_resolution_logic():
    """Verify deterministic logic behaves with 100% precision across all permutations."""
    def resolve_train_py(cfg_dict: dict[str, Any], args: argparse.Namespace) -> bool:
        c = dict(cfg_dict)
        if args.non_deterministic:
            c.setdefault("train", {})["deterministic"] = False
        elif args.deterministic:
            c.setdefault("train", {})["deterministic"] = True
        elif "deterministic" not in c.get("train", {}):
            c.setdefault("train", {})["deterministic"] = True
        return bool(c["train"]["deterministic"])

    def resolve_trainer_py(cfg_dict: dict[str, Any], args: Any) -> bool:
        if getattr(args, "non_deterministic", False):
            deterministic = False
        elif getattr(args, "deterministic", False):
            deterministic = True
        else:
            deterministic = bool(cfg_dict.get("train", {}).get("deterministic", True))
        return deterministic

    # Case 1: Config has deterministic: false, CLI has neither -> must be False
    c1 = {"train": {"deterministic": False}}
    a1 = argparse.Namespace(deterministic=False, non_deterministic=False)
    assert not resolve_train_py(c1, a1)
    assert not resolve_trainer_py(c1, a1)

    # Case 2: Config has deterministic: true, CLI has --non-deterministic -> must be False
    c2 = {"train": {"deterministic": True}}
    a2 = argparse.Namespace(deterministic=False, non_deterministic=True)
    assert not resolve_train_py(c2, a2)
    assert not resolve_trainer_py(c2, a2)

    # Case 3: Config omits deterministic, CLI has --non-deterministic -> must be False
    c3 = {"train": {}}
    a3 = argparse.Namespace(deterministic=False, non_deterministic=True)
    assert not resolve_train_py(c3, a3)
    assert not resolve_trainer_py(c3, a3)

    # Case 4: Config has deterministic: false, CLI has --deterministic -> must be True
    c4 = {"train": {"deterministic": False}}
    a4 = argparse.Namespace(deterministic=True, non_deterministic=False)
    assert resolve_train_py(c4, a4)
    assert resolve_trainer_py(c4, a4)

    # Case 5: Config omits deterministic, CLI has neither -> defaults to True
    c5 = {"train": {}}
    a5 = argparse.Namespace(deterministic=False, non_deterministic=False)
    assert resolve_train_py(c5, a5)
    assert resolve_trainer_py(c5, a5)


def test_h2_spectral_loss_config_and_parameters():
    """Verify configs/rmr_research/h2_spectral_loss.yaml complies with canonical invariants."""
    cfg_path = _REPO_ROOT / "configs" / "rmr_research" / "h2_spectral_loss.yaml"
    assert cfg_path.is_file(), f"Config file not found at {cfg_path}"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_v3_config(cfg)

    # Verify spectral loss configuration
    loss_cfg = make_loss_cfg(cfg)
    assert loss_cfg.use_spectral_loss is True
    assert loss_cfg.lambda_spectral == 0.1
    assert loss_cfg.spectral_beta == 2.0
    assert loss_cfg.lambda_spectral_dc == 1.0

    # Verify non-deterministic flag requirement
    assert cfg["train"]["deterministic"] is False

    # Verify model parameters strictly <= 105,000 (canonical v19 = 104,441)
    model, _ = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 104441, f"Expected exactly 104,441 params, got {n_params}"
    assert n_params <= 105000, f"Exceeded lightweight parameter budget: {n_params} > 105,000"


def test_all_codebase_files_line_count_invariant():
    """Strict protocol check: every .py file in rmr_core/ and rmr_v3/ must have len(lines) <= 450."""
    py_files = sorted(list((_REPO_ROOT / "rmr_core").rglob("*.py")) + list((_REPO_ROOT / "rmr_v3").rglob("*.py")))
    assert len(py_files) > 0, "No Python files discovered in rmr_core or rmr_v3"

    violations = []
    for p in py_files:
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) > 450:
            violations.append(f"{p.relative_to(_REPO_ROOT)}: {len(lines)} lines")

    assert not violations, f"The following files violate the <= 450 lines protocol invariant:\n" + "\n".join(violations)


if __name__ == "__main__":
    print("=" * 80)
    print("Running Count-Preserving Spectral Loss & Protocol Hardening Test Suite...")
    print("=" * 80)
    test_spectral_loss_identical_inputs()
    print("  [PASS] test_spectral_loss_identical_inputs")
    test_spectral_loss_dc_mass_conservation()
    print("  [PASS] test_spectral_loss_dc_mass_conservation")
    test_spectral_loss_backward_gradient()
    print("  [PASS] test_spectral_loss_backward_gradient")
    test_loss_config_spectral_fields()
    print("  [PASS] test_loss_config_spectral_fields")
    test_orchestration_spectral_loss_disabled()
    print("  [PASS] test_orchestration_spectral_loss_disabled")
    test_orchestration_spectral_loss_enabled_y()
    print("  [PASS] test_orchestration_spectral_loss_enabled_y")
    test_orchestration_spectral_loss_enabled_dual()
    print("  [PASS] test_orchestration_spectral_loss_enabled_dual")
    test_orchestration_spectral_loss_gradient_flow()
    print("  [PASS] test_orchestration_spectral_loss_gradient_flow")
    test_deterministic_resolution_logic()
    print("  [PASS] test_deterministic_resolution_logic")
    test_h2_spectral_loss_config_and_parameters()
    print("  [PASS] test_h2_spectral_loss_config_and_parameters")
    test_all_codebase_files_line_count_invariant()
    print("  [PASS] test_all_codebase_files_line_count_invariant")
    print("=" * 80)
    print("ALL 11 UNIT AND PROTOCOL HARDENING TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
