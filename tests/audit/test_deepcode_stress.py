from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.operators import (
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
)
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    curvature_power_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


def test_empty_regions_resilience():
    """Verify that build_multiscale_regions handles empty region sets without IndexError."""
    reg = build_multiscale_regions(
        height=32,
        width=32,
        output_stride=4,
        region_sizes_px=(),
        include_full_image=False,
    )
    assert reg.boxes.shape == (0, 4)
    assert reg.scale_id.shape == (0,)
    assert reg.area.shape == (0,)

    # Duality and sum operators with 0 boxes must not crash
    x = torch.randn(2, 1, 32, 32)
    s = regional_sum(x, reg.boxes)
    assert s.shape == (2, 1, 0)

    val = torch.randn(2, 1, 0)
    adj = regional_adjoint(val, reg.boxes, height=32, width=32)
    assert adj.shape == (2, 1, 32, 32)
    assert torch.all(adj == 0.0)


def test_regional_sum_and_adjoint_dimension_invariance():
    """Verify regional_sum supports 2D [H,W], 3D [B,H,W], 4D [B,C,H,W] and adjoint duality holds."""
    reg = build_multiscale_regions(16, 16, output_stride=4, region_sizes_px=(16, 32), include_full_image=False)
    m = len(reg.boxes)

    # 4D input
    x_4d = torch.randn(2, 1, 16, 16, dtype=torch.float64)
    s_4d = regional_sum(x_4d, reg.boxes)
    assert s_4d.shape == (2, 1, m)

    # 3D input
    x_3d = x_4d.squeeze(1)
    s_3d = regional_sum(x_3d, reg.boxes)
    assert s_3d.shape == (2, 1, m)
    assert torch.allclose(s_4d, s_3d, atol=1e-8)

    # 2D input
    x_2d = x_4d[0, 0]
    s_2d = regional_sum(x_2d, reg.boxes)
    assert s_2d.shape == (1, 1, m)
    assert torch.allclose(s_4d[0:1], s_2d, atol=1e-8)

    # Adjoint duality: <A x, y> == <x, A^T y>
    y_val = torch.randn(2, 1, m, dtype=torch.float64)
    at_y = regional_adjoint(y_val, reg.boxes, 16, 16)
    lhs = (s_4d * y_val).sum()
    rhs = (x_4d * at_y).sum()
    assert torch.allclose(lhs, rhs, atol=1e-6)

    # 2D values in regional_adjoint [B, M]
    y_2d = y_val.squeeze(1)
    at_y_from_2d = regional_adjoint(y_2d, reg.boxes, 16, 16)
    assert torch.allclose(at_y, at_y_from_2d, atol=1e-8)


def test_losses_zero_cpu_gpu_sync_and_3d_target_support():
    """Verify compute_rmr_v3_losses supports 3D target_y and zero-sync empty masks without NaN."""
    h, w = 32, 32
    reg = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64), include_full_image=False)
    m = len(reg.boxes)

    y = torch.rand(2, 1, h, w, requires_grad=True)
    y0 = torch.rand(2, 1, h, w, requires_grad=True)
    b_reg = torch.rand(2, 1, m, requires_grad=True)
    r_disp = torch.full((2, 1, m), 50.0)

    outputs = {
        "y": y,
        "y0": y0,
        "b_region": b_reg,
        "region_dispersion": r_disp,
        "regions": reg,
    }

    # 4D target
    target_4d = torch.randint(0, 3, (2, 1, h, w)).float()
    cfg = RMRv3LossConfig(dm_target="dual", elementwise_dense_scaling=True)
    losses_4d = compute_rmr_v3_losses(outputs, target_4d, cfg)

    # 3D target
    target_3d = target_4d.squeeze(1)
    losses_3d = compute_rmr_v3_losses(outputs, target_3d, cfg)

    assert torch.allclose(losses_4d["total"], losses_3d["total"], atol=1e-6)

    # Stress test zero-sync: curvature loss when threshold is extremely high (all pixels masked)
    curv_zero = curvature_power_loss(y, target_4d, threshold=1e6, mode="hard")
    assert curv_zero.item() == 0.0
    assert torch.isfinite(curv_zero)

    # Scale alignment when foreground mask is completely empty (target == 0 everywhere)
    empty_target = torch.zeros(2, 1, h, w)
    scale_w = torch.softmax(torch.randn(2, 3, h, w), dim=1)
    align_zero = physical_scale_alignment_loss(scale_w, empty_target, tau_sparse=0.03, mask_background=True)
    assert align_zero.item() == 0.0
    assert torch.isfinite(align_zero)


def test_topk_hard_bg_bounds():
    """Verify topk_hard_background_loss handles edge-case ratios and pure-foreground without crashing."""
    y = torch.rand(2, 1, 16, 16, requires_grad=True)

    # Case 1: Pure crowd (target > bg_threshold everywhere -> 0 background pixels)
    target_crowd = torch.full((2, 1, 16, 16), 5.0)
    l_crowd = topk_hard_background_loss(y, target_crowd)
    assert l_crowd.item() == 0.0
    l_crowd.backward()
    assert y.grad is not None

    # Case 2: ratio > 1.0 (clamped safely to num_bg)
    y.grad = None
    target_bg = torch.zeros(2, 1, 16, 16)
    l_over = topk_hard_background_loss(y, target_bg, ratio=2.0)
    assert torch.isfinite(l_over)
    l_over.backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()

    # Case 3: Exactly 1 background pixel
    target_single = torch.full((1, 1, 4, 4), 10.0)
    target_single[0, 0, 0, 0] = 0.0  # 1 background pixel
    l_single = topk_hard_background_loss(y[:1, :, :4, :4], target_single, ratio=0.01)
    assert torch.isfinite(l_single)


def test_barzilai_borwein_stability_adversarial():
    """Verify Barzilai-Borwein adaptive step size is strictly bounded under adversarial conditions."""
    h, w = 16, 16
    reg = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), include_full_image=False)
    m = len(reg.boxes)

    # Adversarial test 1: Identical iterations (residual difference == 0)
    y0 = torch.zeros(1, 1, h, w)
    b_solver = torch.zeros(1, 1, m)
    weight_solver = torch.ones(1, 1, m)

    res = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=reg,
        iterations=6,
        omega=1.0,
        use_barzilai_borwein=True,
        proximal_mode="firm",
        proximal_tau=0.015,
    )
    assert torch.isfinite(res["y"]).all()
    assert len(res["iterates"]) == 7

    # Adversarial test 2: High gradient spikes
    y_spike = torch.rand(1, 1, h, w) * 100.0
    b_spike = torch.rand(1, 1, m) * 1000.0
    res_spike = unrolled_sirt_solver(
        y0=y_spike,
        b_solver=b_spike,
        weight_solver=weight_solver,
        regions=reg,
        iterations=6,
        omega=1.0,
        use_barzilai_borwein=True,
        proximal_mode="firm",
        proximal_tau=0.015,
        tv_lambda=0.02,
    )
    assert torch.isfinite(res_spike["y"]).all()
    assert not torch.isnan(res_spike["y"]).any()


def test_extreme_crowd_and_empty_background_amp():
    """Verify forward and backward passes with 15,000 count and 0 count under AMP float16 and float32."""
    cfg_yaml = yaml.safe_load(Path("configs/rmr_v21/rmr_v21_canonical.yaml").read_text(encoding="utf-8"))
    model_cfg = RMRv3Config.from_dict(cfg_yaml["model"], pretrained=False)
    model = RMRv3(model_cfg)
    model.train()

    loss_cfg = RMRv3LossConfig.from_dict(cfg_yaml["loss"])

    # Mega-stadium batch: sample 0 has 15,000 count, sample 1 has 0 count (empty background)
    h, w = 128, 128
    img = torch.randn(2, 3, 512, 512)
    target_y = torch.zeros(2, 1, h, w)
    target_y[0, :, :64, :64] = 15000.0 / (64 * 64)  # 15,000 count in top-left quadrant
    # sample 1 remains strictly 0.0

    # Test float32
    out_f32 = model(img)
    losses_f32 = compute_rmr_v3_losses(out_f32, target_y, loss_cfg)
    assert torch.isfinite(losses_f32["total"])
    losses_f32["total"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None gradient in float32"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} gradient contains NaN/Inf in float32"

    # Test AMP autocast
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=True):
        out_amp = model(img)
        losses_amp = compute_rmr_v3_losses(out_amp, target_y, loss_cfg)
        assert torch.isfinite(losses_amp["total"])
    losses_amp["total"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None gradient in AMP"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} gradient contains NaN/Inf in AMP"


def test_odd_prime_canvas_dimensions_end_to_end():
    """Verify that arbitrary odd dimensions (e.g. 173x227) pass through forward and backward cleanly."""
    cfg_yaml = yaml.safe_load(Path("configs/rmr_v21/rmr_v21_canonical.yaml").read_text(encoding="utf-8"))
    model_cfg = RMRv3Config.from_dict(cfg_yaml["model"], pretrained=False)
    model = RMRv3(model_cfg)
    model.train()

    loss_cfg = RMRv3LossConfig.from_dict(cfg_yaml["loss"])
    # Set non-strict DM for arbitrary dimensions
    loss_cfg.dm_strict = False

    img_odd = torch.randn(1, 3, 227, 173)
    out_odd = model(img_odd)

    _, _, gh, gw = out_odd["y"].shape
    target_odd = torch.randint(0, 2, (1, 1, gh, gw)).float()

    losses = compute_rmr_v3_losses(out_odd, target_odd, loss_cfg)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} missing gradient on odd image"
            assert torch.isfinite(p.grad).all()


def test_gradient_flow_completeness_v21():
    """Verify that 100% of trainable parameters in RMR-v21 canonical receive non-zero gradients."""
    cfg_yaml = yaml.safe_load(Path("configs/rmr_v21/rmr_v21_canonical.yaml").read_text(encoding="utf-8"))
    model_cfg = RMRv3Config.from_dict(cfg_yaml["model"], pretrained=False)
    model = RMRv3(model_cfg)
    model.train()

    # Verify hard parameter budget
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 105000, f"Parameter count {total_params} exceeds 105,000 budget!"
    assert total_params == 104441

    loss_cfg = RMRv3LossConfig.from_dict(cfg_yaml["loss"])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    img = torch.randn(2, 3, 512, 512)
    target_y = torch.randint(0, 4, (2, 1, 128, 128)).float()

    # Step 0: Initial backward pass
    out = model(img)
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg)
    losses["total"].backward()

    # Step 0: Ensure router pointwise conv receives active gradients
    assert model.scale_router.pw.weight.grad is not None
    assert model.scale_router.pw.weight.grad.abs().sum().item() > 0.0

    # Step 1: Update weights with gradient clipping (matching production training loop)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    out2 = model(img)
    losses2 = compute_rmr_v3_losses(out2, target_y, loss_cfg)
    losses2["total"].backward()

    zero_grad_params = []
    none_grad_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                none_grad_params.append(name)
            elif p.grad.abs().sum().item() == 0.0:
                zero_grad_params.append(name)

    assert not none_grad_params, f"Parameters with None gradients: {none_grad_params}"
    assert not zero_grad_params, f"Parameters with exactly zero gradients (dead weights): {zero_grad_params}"
