from __future__ import annotations

import torch
import pytest

from rmr_core.necks.rep_aspp import RepASPPLiteNeck
from rmr_v3.model import RMRv3, RMRv3Config


def test_rep_aspp_neck_shapes():
    """Verify shape preservation across clean and arbitrary resolutions."""
    neck = RepASPPLiteNeck(in_channels=(16, 32, 48), width=32, aspp_dilations=(1, 3, 6), use_aspp_gap=True)
    neck.eval()

    # Test clean power-of-two resolution (512x512 input -> C4: 128x128, C8: 64x64, C16: 32x32)
    c4 = torch.randn(2, 16, 128, 128)
    c8 = torch.randn(2, 32, 64, 64)
    c16 = torch.randn(2, 48, 32, 32)
    with torch.no_grad():
        p4, p8, p16 = neck(c4, c8, c16)

    assert p4.shape == (2, 32, 128, 128)
    assert p8.shape == (2, 32, 64, 64)
    assert p16.shape == (2, 32, 32, 32)

    # Test arbitrary odd resolution (409x513 input -> C4: 103x129, C8: 52x65, C16: 26x33)
    c4_odd = torch.randn(1, 16, 103, 129)
    c8_odd = torch.randn(1, 32, 52, 65)
    c16_odd = torch.randn(1, 48, 26, 33)
    with torch.no_grad():
        p4_odd, p8_odd, p16_odd = neck(c4_odd, c8_odd, c16_odd)

    assert p4_odd.shape == (1, 32, 103, 129)
    assert p8_odd.shape == (1, 32, 52, 65)
    assert p16_odd.shape == (1, 32, 26, 33)


def test_rep_aspp_neck_switch_to_deploy_equivalence():
    """Verify algebraic equivalence of structural re-parameterization."""
    neck = RepASPPLiteNeck(in_channels=(16, 32, 48), width=32, aspp_dilations=(1, 3, 6), use_aspp_gap=True)
    neck.eval()

    c4 = torch.randn(1, 16, 64, 64)
    c8 = torch.randn(1, 32, 32, 32)
    c16 = torch.randn(1, 48, 16, 16)

    with torch.no_grad():
        p4_train, p8_train, p16_train = neck(c4, c8, c16)

    neck.switch_to_deploy()
    assert neck.ref16.is_deployed
    assert neck.ref8.is_deployed
    assert neck.ref4.is_deployed

    with torch.no_grad():
        p4_deploy, p8_deploy, p16_deploy = neck(c4, c8, c16)

    max_diff_16 = (p16_train - p16_deploy).abs().max().item()
    max_diff_8 = (p8_train - p8_deploy).abs().max().item()
    max_diff_4 = (p4_train - p4_deploy).abs().max().item()

    assert max_diff_16 < 1e-4, f"P16 deploy diff too high: {max_diff_16}"
    assert max_diff_8 < 1e-4, f"P8 deploy diff too high: {max_diff_8}"
    assert max_diff_4 < 1e-4, f"P4 deploy diff too high: {max_diff_4}"


def test_rmr_v3_with_rep_aspp_neck_param_budget():
    """Verify end-to-end integration and exact parameter ceiling adherence."""
    cfg = RMRv3Config(
        neck_type="rep_aspp_lite",
        feature_width=32,
        aspp_dilations=(1, 3, 6),
        use_aspp_gap=True,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 104441, f"Param ceiling violated: {total_params} > 104,441"
    assert total_params == 103875, f"Expected exactly 103,875 params, got {total_params}"

    # Forward pass check
    x = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.y.shape[-2:] == (64, 64)
    assert out.y0.shape[-2:] == (64, 64)
    assert torch.isfinite(out.y).all()


def test_rmr_v3_track_b_rep_aspp_param_budget():
    """Verify Track B (W=64) integration and <= 200,000 parameter budget."""
    cfg = RMRv3Config(
        neck_type="rep_aspp_lite",
        feature_width=64,
        max_trainable_params=200000,
        aspp_dilations=(1, 3, 6),
        use_aspp_gap=True,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 200000, f"Track B ceiling violated: {total_params} > 200,000"
    assert total_params == 121731, f"Expected 121,731 params for Track B W=64, got {total_params}"

    x = torch.randn(1, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.y.shape[-2:] == (64, 64)
    assert torch.isfinite(out.y).all()

