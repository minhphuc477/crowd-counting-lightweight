import yaml
import pytest
import torch
import torch.nn.functional as F

from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import RMRv3LossConfig
from rmr_v3.reporting import TRAIN_LOG_FIELDNAMES


def test_count_l1_gradient_constancy():
    """Verify that unnormalized count L1 produces constant gradient magnitude regardless of crowd size."""
    # Sparse crowd: N = 20
    pred_sparse = torch.tensor([15.0], requires_grad=True)
    target_sparse = torch.tensor([20.0])
    loss_sparse = F.l1_loss(pred_sparse, target_sparse)
    loss_sparse.backward()
    grad_sparse = pred_sparse.grad.item()

    # Dense crowd: N = 2000
    pred_dense = torch.tensor([1500.0], requires_grad=True)
    target_dense = torch.tensor([2000.0])
    loss_dense = F.l1_loss(pred_dense, target_dense)
    loss_dense.backward()
    grad_dense = pred_dense.grad.item()

    assert abs(grad_sparse) == pytest.approx(1.0)
    assert abs(grad_dense) == pytest.approx(1.0)
    assert grad_sparse == grad_dense  # Both equal -1.0


def test_loss_config_count_l1_validation():
    """Verify lambda_count_l1 schema validation in RMRv3LossConfig."""
    cfg = RMRv3LossConfig(lambda_count_l1=0.5)
    assert cfg.lambda_count_l1 == 0.5

    d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    cfg2 = RMRv3LossConfig.from_dict(d)
    assert cfg2.lambda_count_l1 == 0.5

    with pytest.raises(ValueError, match="lambda_count_l1 must be non-negative"):
        RMRv3LossConfig(lambda_count_l1=-0.1)


def test_reporting_fields_include_count_l1():
    """Verify TRAIN_LOG_FIELDNAMES includes train_count_l1."""
    assert "train_count_l1" in TRAIN_LOG_FIELDNAMES


def test_breakthrough_config_and_parameters():
    """Verify sub60_dense_breakthrough.yaml validates and has exactly 104,441 parameters."""
    with open("configs/rmr_sub60/sub60_dense_breakthrough.yaml") as f:
        cfg = yaml.safe_load(f)

    validate_v3_config(cfg)
    model, _ = make_model(cfg)
    loss_cfg = make_loss_cfg(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 104441, f"Expected exactly 104,441 parameters, got {n_params}"
    assert loss_cfg.lambda_count_l1 == 0.5
    assert cfg["data"]["crop_size"] == 256
    assert cfg["train"]["batch_size"] == 16
    assert cfg["model"]["asymmetric_morozov"] is True
    assert cfg["model"]["resonant_adjoint"] is True


def test_all_sub60_suite_configs_and_parameters():
    """Verify all 4 sub60 suite configs validate and produce exactly 104,441 parameters."""
    cfgs = [
        "configs/rmr_sub60/sub60_abl_crop256_only.yaml",
        "configs/rmr_sub60/sub60_abl_count_l1_only.yaml",
        "configs/rmr_sub60/sub60_abl_crop256_plus_count_l1.yaml",
        "configs/rmr_sub60/sub60_dense_breakthrough.yaml",
    ]
    for path in cfgs:
        with open(path) as f:
            c = yaml.safe_load(f)
        validate_v3_config(c)
        m, _ = make_model(c)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert n == 104441, f"{path} expected 104,441 params, got {n}"
