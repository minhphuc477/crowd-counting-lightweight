from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import torch
import yaml

from rmr_core.losses import balanced_smooth_l1
from rmr_core.operators import build_multiscale_regions, regional_sum
from rmr_v3.eval import load_model_from_ckpt
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config


def test_balanced_smooth_l1_sample_isolation():
    """Verify strict sample isolation in balanced_smooth_l1 under asymmetric crowd density."""
    torch.manual_seed(42)
    # Sample 0: Sparse (10 positive cells out of 1024)
    # Sample 1: Dense (800 positive cells out of 1024)
    target_0 = torch.zeros(1, 1, 32, 32)
    target_0[0, 0, :2, :5] = 1.0  # 10 positive cells

    target_1 = torch.zeros(1, 1, 32, 32)
    target_1[0, 0, :25, :32] = 2.0  # 800 positive cells

    target_batch = torch.cat([target_0, target_1], dim=0)

    pred_batch = torch.zeros(2, 1, 32, 32, requires_grad=True)

    loss_batch = balanced_smooth_l1(pred_batch, target_batch)
    loss_batch.backward()

    # Isolate sample 0 alone
    pred_0 = torch.zeros(1, 1, 32, 32, requires_grad=True)
    loss_0 = balanced_smooth_l1(pred_0, target_0)
    loss_0.backward()

    # Gradient on sample 0 in batch should equal 0.5 * grad on sample 0 alone (due to 1/B averaging)
    torch.testing.assert_close(pred_batch.grad[0:1], 0.5 * pred_0.grad, atol=1e-6, rtol=1e-5)


def test_eval_config_override(tmp_path: Path):
    """Verify that --config YAML override in eval.py successfully overrides checkpoint configuration."""
    model_cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False)
    model = RMRv3(model_cfg)
    ckpt_path = tmp_path / "test_ckpt.pt"
    ckpt_dict = {
        "model": model.state_dict(),
        "config": {
            "model": {"feature_width": 32, "output_stride": 4},
            "eval": {"density_bins": [100.0, 500.0]},
        },
    }
    torch.save(ckpt_dict, ckpt_path)

    override_yaml = tmp_path / "override.yaml"
    override_data = {
        "eval": {"density_bins": [50.0, 300.0]},
    }
    with open(override_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(override_data, f)

    device = torch.device("cpu")
    loaded_model, uniform_rel, loaded_cfg, _ = load_model_from_ckpt(
        ckpt_path, device, use_ema=False, config_path=override_yaml
    )

    assert loaded_cfg["eval"]["density_bins"] == [50.0, 300.0]
    assert loaded_cfg["model"]["feature_width"] == 32


def test_dual_dm_components_logging():
    """Verify that dual dm_target correctly blends dm_components from y and y0."""
    h, w = 32, 32
    torch.manual_seed(42)
    y = torch.rand(1, 1, h, w, requires_grad=True)
    y0 = torch.rand(1, 1, h, w, requires_grad=True)
    target = torch.rand(1, 1, h, w)
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    b_reg = regional_sum(target, regions.boxes)
    disp = torch.full_like(b_reg, 50.0)

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": b_reg,
        "region_dispersion": disp,
    }

    cfg_dual = RMRv3LossConfig(
        dm_target="dual",
        use_multiscale_dm=True,
        dm_block_sizes_px=(16, 32),
        dm_weights=(0.5, 0.5),
        dm_kappas=(20.0, 20.0),
        lambda_count=0.0,
        lambda_cell=0.0,
        lambda_region_nb=0.0,
    )

    losses = compute_rmr_v3_losses(outputs, target, cfg_dual)
    assert "dm_16" in losses
    assert "dm_32" in losses
    assert "allocation_y" in losses
    assert "allocation_y0" in losses

    # Loss allocation must equal exactly 0.5 * (allocation_y + allocation_y0)
    expected_alloc = 0.5 * (losses["allocation_y"] + losses["allocation_y0"])
    torch.testing.assert_close(losses["allocation"], expected_alloc, atol=1e-6, rtol=1e-5)


def test_odd_prime_resolution_gradient_backprop():
    """Verify that the full RMR model handles odd/prime resolutions with finite gradient flow."""
    model_cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False)
    model = RMRv3(model_cfg)
    x = torch.randn(1, 3, 113, 227, requires_grad=True)
    out = model(x)
    assert out.y.shape[-2:] == (29, 57)
    loss = out.y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
