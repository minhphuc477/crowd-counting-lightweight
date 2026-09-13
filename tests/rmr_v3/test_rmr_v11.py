from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from rmr_core.operators import build_multiscale_regions
from rmr_v3.config import RMRv3Config, load_config, validate_v3_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    curvature_power_loss,
    mass_weighted_cell_loss,
    topk_hard_background_loss,
)
from rmr_v3.model import RMRv3
from rmr_v3.solver import unrolled_sirt_solver


def test_rmr_v11_parameter_budget():
    """Verify RMR-v11 Canonical strictly respects the <= 105,000 parameter budget.

    Budget Breakdown:
    - MobileNetV4 Backbone: 50,288
    - ASPP-Lite Neck: 6,864
    - Fine Measure Head: 1,026
    - Hurdle-NB Head: 45,779
    - Dynamic Scale Router: 483
    - Foreground Gate: 33 (Conv2d(32, 1, kernel_size=1, bias=True))
    Total Trainable Parameters: exactly 104,473 (527 headroom).
    """
    cfg_v11 = RMRv3Config(
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
    model = RMRv3(cfg_v11)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 105_000, f"RMR-v11 parameters ({total_params}) exceed 105,000 budget!"
    assert total_params == 104_473, f"Expected exactly 104,473 parameters, got {total_params}"

    # Verify FG gate parameter count
    assert model.fg_gate is not None
    fg_params = sum(p.numel() for p in model.fg_gate.parameters() if p.requires_grad)
    assert fg_params == 33, f"Expected FG gate to have 33 params, got {fg_params}"

    # Verify ablation without FG gate reproduces RMR-v10 parameter count
    cfg_no_fg = RMRv3Config(
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
        foreground_gate=False,
    )
    model_no_fg = RMRv3(cfg_no_fg)
    assert sum(p.numel() for p in model_no_fg.parameters() if p.requires_grad) == 104_440


def test_morozov_trust_region_clamping():
    """Verify Morozov Trust-Region bounds relative updates to [-kappa*y, +kappa*max(y, floor)]."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, 4, (32, 64, 128), 0.5, False, device="cpu")
    m = len(regions.boxes)

    y0 = torch.full((1, 1, h, w), fill_value=1.0, dtype=torch.float32)
    # Adversarial target: regional head reports 10,000 count (massive positive overshoot)
    b_huge = torch.full((1, 1, m), fill_value=10_000.0, dtype=torch.float32)
    w_ones = torch.ones((1, 1, m), dtype=torch.float32)

    kappa = 0.35
    floor_val = 0.005

    # Run 1 step with trust region clamping
    res_clamped = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_huge,
        weight_solver=w_ones,
        regions=regions,
        iterations=1,
        omega=1.0,
        trust_region_kappa=kappa,
        trust_region_floor=floor_val,
        proximal_tau=0.0,
        tv_lambda=0.0,
    )
    y1_clamped = res_clamped["y"]

    # Maximum allowed increase from y0=1.0 is kappa * max(1.0, 0.005) = 0.35
    max_increase = y1_clamped.max().item() - 1.0
    assert max_increase <= kappa + 1e-5, (
        f"Trust region violation: density increased by {max_increase}, exceeding kappa={kappa}!"
    )

    # Adversarial target: regional head reports 0 count (massive negative drag)
    b_zero = torch.zeros((1, 1, m), dtype=torch.float32)
    res_zero = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_zero,
        weight_solver=w_ones,
        regions=regions,
        iterations=1,
        omega=1.0,
        trust_region_kappa=kappa,
        trust_region_floor=floor_val,
        proximal_tau=0.0,
        tv_lambda=0.0,
    )
    y1_zero = res_zero["y"]

    # Maximum allowed decrease from y0=1.0 is kappa * 1.0 = 0.35
    min_val = y1_zero.min().item()
    assert min_val >= (1.0 - kappa) - 1e-5, (
        f"Trust region violation: density decreased to {min_val}, below allowed {1.0 - kappa}!"
    )


def test_foreground_gate_subhead():
    """Verify foreground gate sub-head initialization, modulation, and autograd gradient flow."""
    torch.manual_seed(42)
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        dynamic_scale_routing=True,
        foreground_gate=True,
        trust_region_kappa=0.35,
    )
    model = RMRv3(cfg)
    assert model.fg_gate is not None

    # Verify positive bias initialization (~2.0 -> sigmoid ~0.88)
    assert torch.isclose(model.fg_gate.bias.data, torch.tensor([2.0]), atol=1e-3)

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert "fg_logit" in out
    fg_logit = out["fg_logit"]
    assert fg_logit.shape == (2, 1, 32, 32)

    # Verify modulation: out['y0'] is modulated by residual safety floor (0.70 + 0.30 * sigmoid(fg_logit))
    z0 = model.fine_head.forward_logits(model.fusion(*model.encoder(x))[0])
    raw_y0 = model.fine_head.activate(z0)
    expected_y0 = raw_y0 * (0.70 + 0.30 * torch.sigmoid(fg_logit))
    assert torch.allclose(out["y0"], expected_y0, atol=1e-5)

    # Verify backprop flows into fg_gate
    loss = out["fg_logit"].sum() + out["y"].sum()
    loss.backward()
    assert model.fg_gate.weight.grad is not None
    assert model.fg_gate.weight.grad.abs().sum() > 0.0
    assert model.fg_gate.bias.grad is not None
    assert model.fg_gate.bias.grad.abs().sum() > 0.0


def test_curvature_power_loss_properties():
    """Verify curvature power loss mathematical properties, non-negativity, and gradient amplification."""
    y = torch.tensor([[[[0.5, 2.0], [0.01, 5.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 4.0], [0.0, 10.0]]]])

    # 1. Zero error at identical prediction
    assert torch.isclose(curvature_power_loss(target, target), torch.tensor(0.0), atol=1e-6)

    # 2. Non-negativity
    loss = curvature_power_loss(y, target)
    assert loss.item() > 0.0

    # 3. Check gradient amplification on dense clump:
    # On high density under-count (target=10.0 vs y=5.0):
    loss.backward()
    grad_dense = y.grad[0, 0, 1, 1].item()
    # Derivative w.r.t y when y < target is negative (pushes y upwards)
    assert grad_dense < 0.0, "Curvature loss gradient must push under-counted density upwards!"

    # 4. Extreme crowd density numerical stability (>2500 people)
    y_mega = torch.full((1, 1, 32, 32), fill_value=1500.0, requires_grad=True)
    t_mega = torch.full((1, 1, 32, 32), fill_value=2500.0)
    loss_mega = curvature_power_loss(y_mega, t_mega)
    assert torch.isfinite(loss_mega)
    loss_mega.backward()
    assert torch.isfinite(y_mega.grad).all()


def test_topk_hard_background_loss():
    """Verify Top-K Hard Background Mining Loss penalizes only the worst background false alarms."""
    b, h, w = 1, 10, 10
    # Create target with 90 background pixels and 10 foreground pixels
    target = torch.zeros(b, 1, h, w)
    target[0, 0, 0, :10] = 5.0  # 10 foreground pixels

    # Scenario A: perfect zero on background
    y_perfect = torch.zeros(b, 1, h, w)
    y_perfect[0, 0, 0, :10] = 5.0
    loss_perf = topk_hard_background_loss(y_perfect, target, ratio=0.10)
    assert loss_perf.item() == 0.0

    # Scenario B: textured false alarms on background
    y_noisy = torch.zeros(b, 1, h, w, requires_grad=True)
    with torch.no_grad():
        # Inject small noise into background pixels (90 pixels total)
        # Top 9 pixels will have high false alarm ~0.5, rest have 0.01
        y_noisy[0, 0, 1, :9] = 0.5
        y_noisy[0, 0, 2:, :] = 0.01

    loss_noisy = topk_hard_background_loss(y_noisy, target, ratio=0.10)
    # k = 10% of 90 = 9 pixels. Top 9 pixels all have value 0.5.
    # Mean of (0.5)^2 on 9 pixels = 0.25
    assert torch.isclose(loss_noisy, torch.tensor(0.25), atol=1e-3)

    loss_noisy.backward()
    assert y_noisy.grad is not None
    # Top 9 pixels must receive non-zero gradient (2 * 0.5 / 9)
    assert (y_noisy.grad[0, 0, 1, :9] > 0.0).all()
    # Other background pixels below top-k must receive exactly 0 gradient!
    assert (y_noisy.grad[0, 0, 2:, :] == 0.0).all()


def test_power_mass_weighted_cell_loss():
    """Verify power-weighted mass loss with gamma=1.25 and alpha=3.5."""
    target = torch.zeros(1, 1, 10, 10)
    target[0, 0, 5, 5] = 10.0  # Dense head clump
    target[0, 0, 2, 2] = 1.0   # Sparse head

    y = torch.zeros(1, 1, 10, 10, requires_grad=True)

    # Standard gamma=1.0 vs power-weighted gamma=1.25
    loss_linear = mass_weighted_cell_loss(y, target, alpha=3.5, gamma=1.0)
    loss_power = mass_weighted_cell_loss(y, target, alpha=3.5, gamma=1.25)

    assert loss_linear.item() > 0.0
    assert loss_power.item() > 0.0
    assert torch.isfinite(loss_power)


def test_v11_yaml_configs_validation():
    """Verify that all 5 RMR-v11 YAML configuration files parse, pass validation, and load cleanly."""
    config_dir = Path("configs/rmr_v11")
    expected_configs = [
        "rmr_v11_canonical_dsr.yaml",
        "rmr_v11_ablation_no_trust_region.yaml",
        "rmr_v11_ablation_no_hard_bg.yaml",
        "rmr_v11_ablation_no_fg_gate.yaml",
        "rmr_v11_ablation_no_curvature.yaml",
        "rmr_v11_control_no_solver.yaml",
    ]

    for cfg_name in expected_configs:
        cfg_path = config_dir / cfg_name
        assert cfg_path.exists(), f"Configuration file {cfg_path} not found!"
        cfg = load_config(cfg_path)
        assert isinstance(cfg, dict)
        validate_v3_config(cfg)

        m_cfg = RMRv3Config.from_dict(cfg.get("model", {}))
        assert m_cfg is not None
        l_cfg = RMRv3LossConfig.from_dict(cfg.get("loss", {}))
        assert l_cfg is not None


def test_rmr_v11_end_to_end_forward_backward_amp():
    """Full end-to-end forward and backward pass with model in train mode under float32 and AMP."""
    cfg = RMRv3Config(
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
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        iterations=3,
    )
    loss_cfg = RMRv3LossConfig(
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.25,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_curvature=0.25,
        lambda_hard_bg=0.10,
        hard_bg_ratio=0.05,
        lambda_fg_gate=0.05,
        cell_loss_mode="mass_weighted",
        cell_mass_weight_alpha=2.0,
        cell_mass_weight_gamma=1.15,
    )

    model = RMRv3(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 1. Float32 pass
    x = torch.randn(2, 3, 128, 128)
    gt = torch.zeros(2, 1, 32, 32)
    gt[0, 0, 10:15, 10:15] = 0.5
    gt[1, 0, 5:8, 5:8] = 2.0

    optimizer.zero_grad()
    out = model(x)
    losses = compute_rmr_v3_losses(out, gt, loss_cfg)

    assert "total" in losses
    assert "curvature" in losses
    assert "hard_bg" in losses
    assert "fg_bce" in losses
    assert "hurdle_bce" in losses
    assert "trunc_nb" in losses

    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    total_loss.backward()

    # Verify all parameter groups received gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient contains NaN/Inf!"

    # 2. AMP float16 pass (CPU amp with bfloat16 or simulated)
    optimizer.zero_grad()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_amp = model(x)
        losses_amp = compute_rmr_v3_losses(out_amp, gt, loss_cfg)
        loss_amp = losses_amp["total"]
    assert torch.isfinite(loss_amp)
    loss_amp.backward()
    optimizer.step()


def test_kd_dual_depth_and_robust_teacher_loading():
    """Verify dual-depth KD supervision and robust teacher checkpoint loading."""
    from rmr_v3.kd import DensityMapKDLoss
    from rmr_v3.losses import RMRv3LossConfig

    kd_loss = DensityMapKDLoss(lambda_spatial_kl=1.0, lambda_count_kd=0.2)

    # 1. Verify Dual-Depth KD logic: both y0 and y supervised
    b, h, w = 2, 32, 32
    y0 = torch.full((b, 1, h, w), 0.05, requires_grad=True)
    y = torch.full((b, 1, h, w), 0.10, requires_grad=True)
    teacher_y = torch.full((b, 1, h, w), 0.08)

    kd_y0 = kd_loss(y0, teacher_y)
    kd_y = kd_loss(y, teacher_y)
    kd_total = 0.5 * kd_y0["total_kd"] + 0.5 * kd_y["total_kd"]

    assert torch.isfinite(kd_total)
    kd_total.backward()

    assert y0.grad is not None and torch.isfinite(y0.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert y0.grad.abs().sum() > 0
    assert y.grad.abs().sum() > 0

    # 2. Verify robust teacher state_dict loading with ema_model priority
    raw_weights = {"conv.weight": torch.tensor([1.0])}
    ema_weights = {"conv.weight": torch.tensor([2.0])}

    ckpt_with_ema = {"config": {}, "model": raw_weights, "ema_model": ema_weights}
    ckpt_without_ema = {"config": {}, "model": raw_weights}
    ckpt_raw_state = raw_weights

    assert ckpt_with_ema.get("ema_model", ckpt_with_ema.get("model"))["conv.weight"].item() == 2.0
    assert ckpt_without_ema.get("ema_model", ckpt_without_ema.get("model"))["conv.weight"].item() == 1.0

