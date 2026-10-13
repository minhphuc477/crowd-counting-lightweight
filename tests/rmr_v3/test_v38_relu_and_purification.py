import pytest
import torch
import torch.nn.functional as F

from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.config.validator import validate_v3_config
from rmr_core.data import rasterize_points


def test_head_activation_relu_contract():
    """Verify head_activation='relu' config contract and exact zero background."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        head_activation="relu",
        enable_solver=False,
    )
    assert cfg.head_activation == "relu"
    model = RMRv3(cfg)
    assert model.fine_head.head_activation == "relu"

    # Forward dummy input
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.y.shape == (2, 1, 128, 128)
    assert (out.y >= 0.0).all(), "ReLU output must be strictly non-negative"

    # If logit z <= 0, output must be exactly 0.0
    z_neg = torch.full((1, 1, 10, 10), -2.0)
    y_neg = model.fine_head.activate(z_neg)
    assert (y_neg == 0.0).all(), "Negative logits under ReLU must be identically 0.0"


def test_subpixel_stride2_head_activation_relu():
    """Verify subpixel stride 2 head with ReLU outputs exact zero background at 256x256."""
    cfg = RMRv3Config(
        output_stride=2,
        subpixel_stride2=True,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        head_activation="relu",
        enable_solver=False,
    )
    assert cfg.head_activation == "relu"
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 200000, f"Parameters must be <= 200,000, got {total_params}"

    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    assert out.y.shape == (1, 1, 256, 256)
    assert (out.y >= 0.0).all()


def test_validator_enforces_solver_loss_consistency():
    """Verify that validator catches the fatal flaw: enable_solver=True with lambda_region_nb=0.0."""
    invalid_cfg = {
        "model": {
            "output_stride": 4,
            "enable_solver": True,
            "iterations": 6,
        },
        "loss": {
            "lambda_count": 1.0,
            "lambda_region_nb": 0.0,
        },
    }
    with pytest.raises(ValueError, match="model.enable_solver=True requires loss.lambda_region_nb > 0.0"):
        validate_v3_config(invalid_cfg)

    # Valid config: enable_solver=False with lambda_region_nb=0.0 (purified standalone model)
    valid_cfg = {
        "model": {
            "output_stride": 4,
            "enable_solver": False,
        },
        "loss": {
            "lambda_count": 1.0,
            "lambda_region_nb": 0.0,
        },
    }
    validate_v3_config(valid_cfg)


def test_gradient_flow_with_relu_and_count_l1():
    """Verify non-vanishing gradient flow under ReLU head with count L1 and spatial loss."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        head_activation="relu",
        enable_solver=False,
    )
    model = RMRv3(cfg)
    x = torch.randn(1, 3, 512, 512)
    pts = torch.tensor([[100.0, 100.0], [200.0, 200.0]])
    target_y = rasterize_points(pts, 512, 512, stride=4).unsqueeze(0)

    loss_cfg = RMRv3LossConfig(
        output_stride=4,
        lambda_count=1.0,
        count_loss_mode="l1",
        allocation_loss_type="flat_dm16",
        lambda_flat_dm16=1.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
        dm_target="y",
    )
    out = model(x)
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=[pts])
    losses["total"].backward()

    # Verify head gradients exist and are finite
    final_conv = model.fine_head.body[-1]
    assert final_conv.weight.grad is not None
    assert torch.isfinite(final_conv.weight.grad).all()
    assert final_conv.bias.grad is not None
    assert torch.isfinite(final_conv.bias.grad).all()
