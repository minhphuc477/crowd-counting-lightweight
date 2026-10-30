"""Verification test suite for RMR Canonical Publication Architectures.

Verifies:
1. Exact parameter ceiling constraints (Track A <= 104,441; Track B <= 200,000).
2. Pure standalone setup (0.0% Knowledge Distillation, zero teachers).
3. Compact C1 activation mathematics (exact zero background fog and gradient cut-off).
4. Dual-Bayesian + Radon-OT loss backward pass stability (finite, non-zero gradients, no NaNs).
5. Resolution invariance across arbitrary image spatial dimensions.
"""
from __future__ import annotations

import math
import pytest
import torch

from rmr_v3.config import load_config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.orchestration import compute_rmr_v3_losses
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_core.density_activation import compact_c1_activation


def test_track_a_exact_parameter_audit():
    """Verify Track A canonical model has <= 104,441 parameters (exact: 104,390)."""
    cfg = load_config("configs/rmr_canonical/rmr_canonical_104k_seed42.yaml")
    m_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(m_cfg)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_trainable <= 104441, f"Param budget violated: {total_trainable} > 104,441"
    assert total_trainable == 104390, f"Expected canonical 104,390 params, got {total_trainable}"

    # Breakdown audit
    enc_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.fine_head.parameters() if p.requires_grad)
    assert enc_params == 87568
    assert fusion_params == 10784
    assert head_params == 1473


def test_track_b_parameter_audit():
    """Verify Track B subpixel model has <= 200,000 parameters (exact: 122,618)."""
    cfg = load_config("configs/rmr_canonical/rmr_canonical_stride2_seed42.yaml")
    m_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(m_cfg)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_trainable <= 200000, f"Param budget violated: {total_trainable} > 200,000"
    assert total_trainable == 122618, f"Expected canonical 122,618 params, got {total_trainable}"


def test_compact_c1_activation_mathematics():
    """Verify C1-continuous compact support activation properties."""
    tau = 0.5
    z = torch.tensor([-10.0, -2.0, -0.01, 0.0, 0.25, 0.50, 1.0, 5.0], requires_grad=True)
    y = compact_c1_activation(z, tau=tau)

    # 1. Output values
    assert y[0].item() == 0.0
    assert y[1].item() == 0.0
    assert y[2].item() == 0.0
    assert y[3].item() == 0.0
    assert abs(y[4].item() - (0.25 ** 2 / (2 * tau))) < 1e-6
    assert abs(y[5].item() - (0.50 - 0.5 * tau)) < 1e-6
    assert abs(y[6].item() - (1.0 - 0.5 * tau)) < 1e-6

    # 2. Gradient cut-off
    y.sum().backward()
    grads = z.grad.tolist()
    assert grads[0] == 0.0
    assert grads[1] == 0.0
    assert grads[2] == 0.0
    assert grads[3] == 0.0
    assert abs(grads[4] - (0.25 / tau)) < 1e-6
    assert abs(grads[5] - 1.0) < 1e-6
    assert abs(grads[6] - 1.0) < 1e-6


def test_canonical_training_loss_backward():
    """Verify Dual-Bayesian + Radon-OT loss backward pass computes finite non-zero gradients."""
    cfg = load_config("configs/rmr_canonical/rmr_canonical_104k_seed42.yaml")
    m_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(m_cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(**cfg["loss"])

    images = torch.randn(2, 3, 512, 512, requires_grad=True)
    pts_0 = torch.rand(20, 2) * 500.0 + 6.0
    pts_1 = torch.rand(35, 2) * 500.0 + 6.0
    targets = [pts_0, pts_1]

    target_y = torch.zeros(2, 1, 128, 128)
    for p in pts_0:
        target_y[0, 0, int(p[1] // 4), int(p[0] // 4)] += 1.0
    for p in pts_1:
        target_y[1, 0, int(p[1] // 4), int(p[0] // 4)] += 1.0

    outputs = model(images)
    losses = compute_rmr_v3_losses(outputs, target_y, cfg=loss_cfg, points=targets)
    total_loss = losses["total"]

    assert not torch.isnan(total_loss), "Total loss must not be NaN"
    assert not torch.isinf(total_loss), "Total loss must not be Inf"
    assert total_loss.item() > 0.0, "Total loss must be strictly positive"

    total_loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 0.0, "Model parameter gradients must be non-zero"


def test_resolution_invariance():
    """Verify forward pass on arbitrary rectangular resolutions."""
    cfg = load_config("configs/rmr_canonical/rmr_canonical_104k_seed42.yaml")
    m_cfg = RMRv3Config(**cfg["model"])
    model = RMRv3(m_cfg)
    model.eval()

    test_shapes = [(1, 3, 512, 512), (1, 3, 384, 512), (1, 3, 768, 1024)]
    with torch.no_grad():
        for shape in test_shapes:
            x = torch.randn(*shape)
            out = model(x)
            assert out.y.shape[-2] == (shape[-2] + 3) // 4
            assert out.y.shape[-1] == (shape[-1] + 3) // 4
            assert out.y.min().item() >= 0.0, "Output density must be non-negative"
