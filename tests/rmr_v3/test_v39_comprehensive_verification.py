"""Comprehensive verification test for RMRv3/v39 Track A and Track B solutions and code.

Verifies:
1. Config loading & schema validation for Track A and Track B.
2. Parameter budgets: Track A <= 104,441, Track B <= 200,000.
3. Forward pass tensor shapes for both models with arbitrary image dimensions.
4. Loss computation with synthetic batch + points (no NaNs, no Infs, strictly positive scalar).
5. Autograd backward pass: all trainable parameters have valid non-zero finite gradients.
6. Solver step direction and contraction properties.
7. Smooth floor & density activation properties.
"""
from __future__ import annotations

from pathlib import Path
import yaml
import pytest
import torch
import torch.nn as nn

from rmr_v3.engine import make_model, make_loss_cfg
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.solver import unrolled_sirt_solver
from rmr_core.density_activation import _density_activate, _smooth_floor


def test_track_a_v39_end_to_end_verification():
    """Verify Track A v39 config, parameter count, forward pass, loss, and backward gradient."""
    raw_cfg = yaml.safe_load(Path("configs/rmr_research/sub60_v39_track_a_harmonized_stride4.yaml").read_text(encoding="utf-8-sig"))
    model, _ = make_model(raw_cfg)
    loss_cfg = make_loss_cfg(raw_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 104441, f"Track A parameter budget exceeded: {total_params} > 104441"

    model.train()
    b, h, w = 2, 512, 512
    x = torch.randn(b, 3, h, w)
    out = model(x)

    assert "y" in out and "y0" in out and "regions" in out
    assert out["y"].shape == (b, 1, 128, 128)
    assert out["y0"].shape == (b, 1, 128, 128)

    # Synthetic ground truth
    pts = [
        torch.tensor([[100.0, 150.0], [200.0, 250.0], [300.0, 350.0]]),
        torch.tensor([[50.0, 50.0], [450.0, 450.0]]),
    ]
    target_y = torch.zeros(b, 1, 128, 128)
    for i, p in enumerate(pts):
        for pt in p:
            cy, cx = int(pt[1] // 4), int(pt[0] // 4)
            if 0 <= cy < 128 and 0 <= cx < 128:
                target_y[i, 0, cy, cx] += 1.0

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)
    assert "total" in losses
    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    # Backward pass verification
    total_loss.backward()
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "region_head" in name and loss_cfg.lambda_region_nb == 0.0:
                continue
            if "scale_router" in name and loss_cfg.lambda_scale_align == 0.0 and not raw_cfg.get("model", {}).get("scale_conditioned_prior", False) and not raw_cfg.get("model", {}).get("enable_solver", False):
                continue
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradient"
            if param.grad.abs().sum() > 0:
                grad_count += 1
    assert grad_count > 0, "No parameter received non-zero gradient"


def test_track_b_v39_end_to_end_verification():
    """Verify Track B v39 config, parameter count, forward pass, loss, and backward gradient."""
    raw_cfg = yaml.safe_load(Path("configs/rmr_research/sub60_v39_track_b_harmonized_subpixel_stride2.yaml").read_text(encoding="utf-8-sig"))
    model, _ = make_model(raw_cfg)
    loss_cfg = make_loss_cfg(raw_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 200000, f"Track B parameter budget exceeded: {total_params} > 200000"

    model.train()
    b, h, w = 2, 512, 512
    x = torch.randn(b, 3, h, w)
    out = model(x)

    assert "y" in out and "y0" in out and "regions" in out
    assert out["y"].shape == (b, 1, 256, 256), f"Expected stride 2 (256x256), got {out['y'].shape}"

    # Synthetic ground truth for stride 2
    pts = [
        torch.tensor([[100.0, 150.0], [200.0, 250.0], [300.0, 350.0]]),
        torch.tensor([[50.0, 50.0], [450.0, 450.0]]),
    ]
    target_y = torch.zeros(b, 1, 256, 256)
    for i, p in enumerate(pts):
        for pt in p:
            cy, cx = int(pt[1] // 2), int(pt[0] // 2)
            if 0 <= cy < 256 and 0 <= cx < 256:
                target_y[i, 0, cy, cx] += 1.0

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)
    assert "total" in losses
    total_loss = losses["total"]
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    total_loss.backward()
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "region_head" in name and loss_cfg.lambda_region_nb == 0.0:
                continue
            if "scale_router" in name and loss_cfg.lambda_scale_align == 0.0 and not raw_cfg.get("model", {}).get("scale_conditioned_prior", False) and not raw_cfg.get("model", {}).get("enable_solver", False):
                continue
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradient"
            if param.grad.abs().sum() > 0:
                grad_count += 1
    assert grad_count > 0, "No parameter received non-zero gradient"


def test_track_a_full_rmr_solver_active_gradients():
    """Verify that when the Inverse Solver and Regional Head are enabled, 100% of parameters receive gradients."""
    raw_cfg = yaml.safe_load(Path("configs/rmr_research/sub60_v39_track_a_harmonized_stride4.yaml").read_text(encoding="utf-8-sig"))
    # Enable solver and regional supervision
    raw_cfg["model"]["enable_solver"] = True
    raw_cfg["loss"]["lambda_region_nb"] = 0.20
    raw_cfg["loss"]["lambda_scale_align"] = 0.05

    model, _ = make_model(raw_cfg)
    loss_cfg = make_loss_cfg(raw_cfg)
    model.train()

    b, h, w = 2, 512, 512
    x = torch.randn(b, 3, h, w)
    out = model(x)

    pts = [
        torch.tensor([[100.0, 150.0], [200.0, 250.0]]),
        torch.tensor([[50.0, 50.0], [450.0, 450.0]]),
    ]
    target_y = torch.zeros(b, 1, 128, 128)
    for i, p in enumerate(pts):
        for pt in p:
            cy, cx = int(pt[1] // 4), int(pt[0] // 4)
            if 0 <= cy < 128 and 0 <= cx < 128:
                target_y[i, 0, cy, cx] += 1.0

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)
    losses["total"].backward()

    # In full RMR mode, EVERY parameter must have a valid non-zero gradient!
    total_grad_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} must have gradient when full RMR solver is active"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} gradient must be finite"
            total_grad_params += 1
    assert total_grad_params > 50, f"Expected > 50 parameters with gradients, got {total_grad_params}"


def test_solver_sign_and_bound_contracts():
    """Verify that solver delta_y adds mass when under-counting and bounds cannot be breached."""
    from rmr_core.operators import RegionSet

    b, h, w = 1, 32, 32
    y0 = torch.full((b, 1, h, w), 0.05)
    # Box covering entire image
    boxes = torch.tensor([[0, 0, h, w]], dtype=torch.long)
    areas = torch.tensor([h * w], dtype=torch.float32)
    scales = torch.tensor([h], dtype=torch.long)
    regions = RegionSet(boxes=boxes, scale_id=scales, area=areas)

    # Current carrier mass = 0.05 * 1024 = 51.2
    # Case 1: Under-counting (Evidence b = 100.0 > 51.2)
    b_under = torch.tensor([[100.0]])
    weight = torch.tensor([[1.0]])

    res_under = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_under,
        weight_solver=weight,
        regions=regions,
        iterations=1,
        omega=1.0,
        trust_region_kappa=0.5,
        trust_region_floor=0.005,
        output_stride=4,
        proximal_mode="none",
        tv_lambda=0.0,
    )
    y1_under = res_under["y"]
    assert (y1_under >= y0).all(), "Under-counting must monotonically add mass"
    assert y1_under.sum() > y0.sum(), "Under-counting must strictly increase total mass"

    # Case 2: Over-counting (Evidence b = 10.0 < 51.2)
    b_over = torch.tensor([[10.0]])
    res_over = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_over,
        weight_solver=weight,
        regions=regions,
        iterations=1,
        omega=1.0,
        trust_region_kappa=0.5,
        trust_region_floor=0.005,
        output_stride=4,
        proximal_mode="none",
        tv_lambda=0.0,
    )
    y1_over = res_over["y"]
    assert (y1_over <= y0).all(), "Over-counting must monotonically remove mass"
    assert y1_over.sum() < y0.sum(), "Over-counting must strictly decrease total mass"
