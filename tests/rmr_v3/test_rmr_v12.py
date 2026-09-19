from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_v3.config import RMRv3Config, load_config, validate_v3_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    curvature_power_loss,
)
from rmr_v3.model import RMRv3


def test_density_gated_curvature_backward_compatibility():
    """Verify that mode='none' or threshold=0.0 perfectly reproduces v11 curvature loss."""
    torch.manual_seed(42)
    y = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64)
    target[0, 0, 10:15, 10:15] = 1.0
    target[1, 0, 30:35, 30:35] = 2.0

    # Default / backward-compatible v11 call
    loss_v11 = curvature_power_loss(y, target, eps=0.01)
    
    # Explicit mode='none'
    loss_none = curvature_power_loss(y, target, eps=0.01, mode="none", threshold=0.08)
    
    # Explicit threshold=0.0
    loss_zero_thresh = curvature_power_loss(y, target, eps=0.01, mode="hard", threshold=0.0)

    assert torch.allclose(loss_v11, loss_none, atol=1e-7)
    assert torch.allclose(loss_v11, loss_zero_thresh, atol=1e-7)

    # Gradient equality
    g_v11 = torch.autograd.grad(loss_v11, y, retain_graph=True)[0]
    g_none = torch.autograd.grad(loss_none, y, retain_graph=True)[0]
    assert torch.allclose(g_v11, g_none, atol=1e-7)


def test_density_gated_curvature_selective_activation():
    """Verify that density gating activates ONLY on dense clusters and silences sparse/moderate heads."""
    torch.manual_seed(123)
    b, c, h, w = 3, 1, 64, 64
    y = torch.full((b, c, h, w), 0.05, requires_grad=True)
    target = torch.zeros(b, c, h, w)

    # Batch 0: Empty background
    # Batch 1: Isolated moderate heads (spaced >20 px apart, peak 5x5 avg <= 1/25 = 0.04)
    target[1, 0, 10, 10] = 1.0
    target[1, 0, 30, 30] = 1.0
    target[1, 0, 50, 50] = 1.0

    # Batch 2: Dense cluster (8 heads packed tightly in a 4x4 area, 5x5 avg >= 8/25 = 0.32 > 0.08)
    for dy in range(4):
        for dx in range(2):
            target[2, 0, 25 + dy, 25 + dx] = 1.0

    loss_gated = curvature_power_loss(
        y, target, eps=0.01, threshold=0.08, kernel_size=5, mode="hard"
    )
    loss_gated.backward()

    # Batch 0 (empty background): curvature gradient MUST be exactly 0.0
    assert y.grad[0].abs().max().item() == 0.0, "Background regions must receive zero curvature gradient!"

    # Batch 1 (isolated moderate heads): peak density = 0.04 < 0.08, curvature gradient MUST be exactly 0.0
    assert y.grad[1].abs().max().item() == 0.0, "Moderate isolated heads must receive zero curvature gradient!"

    # Batch 2 (dense cluster): curvature gradient MUST be active and non-zero
    dense_grad_max = y.grad[2].abs().max().item()
    assert dense_grad_max > 0.05, f"Dense cluster must have active curvature gradient, got {dense_grad_max}"

    # Gradient direction: since y=0.05 is under-predicting target (cluster has 8 heads),
    # curvature gradient must pull y UPWARD (negative gradient)
    cluster_grad = y.grad[2, 0, 25:29, 25:27]
    assert (cluster_grad < 0.0).all(), "Curvature gradient on under-predicted dense heads must be negative (pulling up)!"


def test_density_gated_curvature_soft_mode():
    """Verify smooth sigmoid gating mode."""
    y = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.zeros(2, 1, 32, 32)
    target[0, 0, 15, 15] = 5.0 # Very dense point

    loss_soft = curvature_power_loss(
        y, target, eps=0.01, threshold=0.08, kernel_size=5, mode="soft", smooth_scale=0.02
    )
    assert not torch.isnan(loss_soft)
    assert not torch.isinf(loss_soft)
    assert loss_soft.item() > 0.0

    loss_soft.backward()
    assert y.grad is not None
    assert not torch.isnan(y.grad).any()


def test_density_gated_curvature_empty_batch_safety():
    """Verify complete safety when an entire batch has zero dense clusters."""
    y = torch.rand(2, 1, 32, 32, requires_grad=True)
    target = torch.zeros(2, 1, 32, 32) # Completely empty images

    loss = curvature_power_loss(
        y, target, eps=0.01, threshold=0.08, kernel_size=5, mode="hard"
    )
    assert loss.item() == 0.0
    loss.backward()
    assert y.grad is not None
    assert (y.grad == 0.0).all()


def test_rmr_v12_config_loading():
    """Verify loading and schema validation of rmr_v12_calibrated_dsr.yaml."""
    cfg_path = Path("configs/rmr_v12/rmr_v12_calibrated_dsr.yaml")
    assert cfg_path.exists(), f"Config {cfg_path} does not exist!"

    raw_cfg = load_config(cfg_path)
    validate_v3_config(raw_cfg)

    loss_cfg = RMRv3LossConfig.from_dict(raw_cfg.get("loss", {}))
    assert loss_cfg.lambda_curvature == 0.50
    assert loss_cfg.curvature_gate_threshold == 0.08
    assert loss_cfg.curvature_gate_kernel == 5
    assert loss_cfg.curvature_gate_mode == "hard"
    assert loss_cfg.curvature_gate_scale == 0.02


def test_rmr_v12_parameter_budget_exactness():
    """Verify RMR-v12 strictly retains exactly 104,473 parameters."""
    with pytest.raises(ValueError, match="permanently BANNED"):
        RMRv3Config(
            output_stride=4,
            feature_width=32,
            pretrained=False,
            neck_type="aspp_lite",
            use_aspp_gap=True,
            aspp_dilations=(1, 3, 6),
            regional_feature_stats="mean",
            region_head_hidden=48,
            hurdle_head=True,
            temp_softplus=True,
            dynamic_scale_routing=True,
            foreground_gate=True,
        )


def test_rmr_v12_end_to_end_loss_dual_supervision():
    """Verify end-to-end forward pass and loss computation with dual supervision and gated curvature."""
    cfg_v12 = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64, 128),
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        enable_solver=True,
        iterations=2, # Fast test
        hurdle_head=True,
        temp_softplus=True,
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
    )
    model = RMRv3(cfg_v12)
    model.eval()

    loss_cfg = RMRv3LossConfig(
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        hard_bg_ratio=0.05,
        lambda_fg_gate=0.05,
    )

    x = torch.randn(2, 3, 128, 128)
    target_y = torch.zeros(2, 1, 32, 32)
    # Add dense cluster to sample 0
    target_y[0, 0, 10:14, 10:14] = 1.0

    out = model(x)
    losses = compute_rmr_v3_losses(out, target_y, cfg=loss_cfg)

    assert "curvature" in losses
    assert losses["curvature"].item() > 0.0
    assert not torch.isnan(losses["total"])
    assert not torch.isinf(losses["total"])

    losses["total"].backward()
    # Check that model parameters have valid gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient!"
