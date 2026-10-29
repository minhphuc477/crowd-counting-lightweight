from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.necks.rlk_hdc import RLKHDCFPNNeck, StaticFastNormalizedFusion
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config


def test_rlk_hdc_training_forward() -> None:
    """Test forward pass shapes across P4, P8, P16."""
    neck = RLKHDCFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    b, h, w = 2, 512, 512
    c4 = torch.randn(b, 16, h // 4, w // 4)
    c8 = torch.randn(b, 32, h // 8, w // 8)
    c16 = torch.randn(b, 48, h // 16, w // 16)

    p4, p8, p16 = neck(c4, c8, c16)

    assert p4.shape == (b, 32, 128, 128)
    assert p8.shape == (b, 32, 64, 64)
    assert p16.shape == (b, 32, 32, 32)
    assert not torch.isnan(p4).any()
    assert not torch.isnan(p8).any()
    assert not torch.isnan(p16).any()


def test_rlk_hdc_reparam_numerical_equivalence() -> None:
    """Verify algebraic re-parameterization equivalence: max |Y_deploy - Y_train| < 1e-5."""
    torch.manual_seed(42)
    neck = RLKHDCFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    neck.eval()

    b, h, w = 2, 256, 256
    c4 = torch.randn(b, 16, h // 4, w // 4)
    c8 = torch.randn(b, 32, h // 8, w // 8)
    c16 = torch.randn(b, 48, h // 16, w // 16)

    with torch.no_grad():
        p4_train, p8_train, p16_train = neck(c4, c8, c16)

    neck.switch_to_deploy()

    with torch.no_grad():
        p4_deploy, p8_deploy, p16_deploy = neck(c4, c8, c16)

    diff4 = torch.max(torch.abs(p4_deploy - p4_train)).item()
    diff8 = torch.max(torch.abs(p8_deploy - p8_train)).item()
    diff16 = torch.max(torch.abs(p16_deploy - p16_train)).item()

    assert diff4 < 1e-5, f"P4 re-param numerical error: {diff4}"
    assert diff8 < 1e-5, f"P8 re-param numerical error: {diff8}"
    assert diff16 < 1e-5, f"P16 re-param numerical error: {diff16}"


def test_rlk_hdc_parameter_budget_track_a() -> None:
    """Verify parameter count satisfies Track A strict ceiling <= 104,441."""
    neck = RLKHDCFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    p_neck_train = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    assert p_neck_train == 10756, f"Expected 10756, got {p_neck_train}"

    neck.eval()
    neck.switch_to_deploy()
    p_neck_deploy = sum(p.numel() for p in neck.parameters() if p.requires_grad)
    assert p_neck_deploy == 10180, f"Expected 10180, got {p_neck_deploy}"

    cfg = RMRv3Config(neck_type="rlk_hdc", subpixel_stride2=False)
    model = RMRv3(cfg)
    total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_train <= 104441, f"Exceeded Track A ceiling: {total_train} > 104,441"
    assert total_train == 103879


def test_rlk_hdc_fast_norm_fusion_convexity() -> None:
    """Verify fast normalized fusion weights are non-negative and convex."""
    fuse = StaticFastNormalizedFusion(eps=1e-4)
    lat = torch.ones(2, 32, 16, 16)
    up = torch.full((2, 32, 16, 16), 3.0)

    # Initial weights: equal (1.0, 1.0) -> expected output is 0.5 * 1 + 0.5 * 3 = 2.0
    out = fuse(lat, up)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-3)

    # Test gradient flow to fusion weights
    loss = out.sum()
    loss.backward()
    assert fuse.weights.grad is not None
    assert not torch.isnan(fuse.weights.grad).any()


def test_rlk_hdc_gradient_flow() -> None:
    """Verify backward pass propagates valid gradients through the full neck."""
    neck = RLKHDCFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    c4 = torch.randn(2, 16, 32, 32, requires_grad=True)
    c8 = torch.randn(2, 32, 16, 16, requires_grad=True)
    c16 = torch.randn(2, 48, 8, 8, requires_grad=True)

    p4, p8, p16 = neck(c4, c8, c16)
    loss = p4.sum() + p8.sum() + p16.sum()
    loss.backward()

    assert c4.grad is not None and not torch.isnan(c4.grad).any()
    assert c8.grad is not None and not torch.isnan(c8.grad).any()
    assert c16.grad is not None and not torch.isnan(c16.grad).any()
    for name, param in neck.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_rlk_hdc_prime_spatial_dimensions() -> None:
    """Verify interpolation parity under irregular and prime spatial dimensions."""
    neck = RLKHDCFPNNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    c4 = torch.randn(2, 16, 67, 73)
    c8 = torch.randn(2, 32, 34, 37)
    c16 = torch.randn(2, 48, 17, 19)

    p4, p8, p16 = neck(c4, c8, c16)
    assert p4.shape == (2, 32, 67, 73)
    assert p8.shape == (2, 32, 34, 37)
    assert p16.shape == (2, 32, 17, 19)


def test_rlk_hdc_half_and_deploy_invariance() -> None:
    """Verify switch_to_deploy works identically in float16 (half) mode."""
    neck_half = RLKHDCFPNNeck().half().eval()
    c4_h = torch.randn(2, 16, 64, 64, dtype=torch.float16)
    c8_h = torch.randn(2, 32, 32, 32, dtype=torch.float16)
    c16_h = torch.randn(2, 48, 16, 16, dtype=torch.float16)

    with torch.no_grad():
        p4_h, p8_h, p16_h = neck_half(c4_h, c8_h, c16_h)

    neck_half.switch_to_deploy()
    with torch.no_grad():
        p4_d, p8_d, p16_d = neck_half(c4_h, c8_h, c16_h)

    diff = torch.max(torch.abs(p4_d - p4_h)).item()
    assert diff < 1e-4, f"Half precision deploy error too high: {diff}"
