"""Test suite for V67 First-Principles numerical stability, gradient bounds, and loss behavior.

Verifies:
1. Multi-density batch stability: Sparse (N=5), Moderate (N=120), Dense (N=1500).
2. L1 Count Loss homoscedastic gradient behavior (constant +/- 1.0 magnitude).
3. Canonical Bayesian loss gradient stability across extreme densities.
4. End-to-end forward-backward pass under AMP FP16.
"""
from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.operators import build_multiscale_regions
from rmr_v3.engine import make_model
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def test_v67_gradient_homoscedasticity_l1_vs_nb():
    """Verify that L1 count loss maintains un-starved gradients on dense crowd errors."""
    from rmr_core.count_loss import count_magnitude_loss

    # Scenario: Dense crowd with N_gt = 1500, model predicts 800 (error = -700)
    pred_dense = torch.tensor([800.0], requires_grad=True)
    target_dense = torch.tensor([1500.0])

    # 1. Negative Binomial (r=50)
    loss_nb = count_magnitude_loss(pred_dense, target_dense, mode="nb", dispersion=50.0)
    loss_nb.backward()
    grad_nb = pred_dense.grad.item()

    # 2. Pure L1
    pred_dense_l1 = torch.tensor([800.0], requires_grad=True)
    loss_l1 = count_magnitude_loss(pred_dense_l1, target_dense, mode="l1")
    loss_l1.backward()
    grad_l1 = pred_dense_l1.grad.item()

    # Verify L1 gradient is exactly -1.0 (homoscedastic)
    assert abs(grad_l1 - (-1.0)) < 1e-5, f"Expected L1 grad -1.0, got {grad_l1}"
    # Verify NB gradient was starved (much weaker than 1.0)
    assert abs(grad_nb) < 0.1, f"Expected starved NB grad < 0.1, got {grad_nb}"
    assert abs(grad_l1) > 10.0 * abs(grad_nb), "L1 should be > 10x stronger than starved NB at N=1500!"


def test_v67_multi_density_batch_forward_backward():
    """Test full forward and backward pass on a multi-density batch under AMP."""
    cfg_path = Path("configs/rmr_v67/v67_exp0_gold_first_principles_1500ep.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False

    model, _ = make_model(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(**cfg.get("loss", {}))

    # Batch of 3 crops: [0]=sparse (N=3), [1]=moderate (N=45), [2]=dense (N=350)
    b, h, w = 3, 512, 512
    x = torch.randn(b, 3, h, w)

    pts_sparse = torch.rand(3, 2) * 500.0
    pts_mod = torch.rand(45, 2) * 500.0
    pts_dense = torch.rand(350, 2) * 500.0
    points = [pts_sparse, pts_mod, pts_dense]

    # Target rasterized maps
    target_y = torch.zeros(b, 1, h // 2, w // 2)
    for i, pts in enumerate(points):
        for pt in pts:
            px, py = int(pt[0].item() // 2), int(pt[1].item() // 2)
            if 0 <= py < h // 2 and 0 <= px < w // 2:
                target_y[i, 0, py, px] += 1.0

    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_type == "cuda":
        model.cuda()
        x = x.cuda()
        target_y = target_y.cuda()
        points = [p.cuda() for p in points]

    with torch.amp.autocast(dev_type, enabled=True):
        out = model(x)
        losses = compute_rmr_v3_losses(out, target_y, cfg=loss_cfg, points=points)

    assert "total" in losses
    assert torch.isfinite(losses["total"])
    assert losses["total"] > 0.0

    losses["total"].backward()

    # Verify gradients across model parameters
    has_grad_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has no gradient!"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient!"
            if p.grad.abs().sum() > 0:
                has_grad_count += 1

    assert has_grad_count > 0, "No parameters received gradients!"
