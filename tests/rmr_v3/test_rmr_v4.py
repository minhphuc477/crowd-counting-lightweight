import math
import pytest
import torch
import torch.nn as nn

from rmr_core.operators import build_multiscale_regions
from rmr_v2.losses import flat_dm16_loss, flat_dm_block_loss, hierarchical_dm_loss
from rmr_v3.config import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import (
    ProbabilisticRegionalEvidenceHead,
    RMRv3,
    RMRv3Config,
    _map_boxes_between_grids,
    region_mean_std_features,
)


def test_native_box_mapping_nonempty():
    src_hw = (128, 128)
    dst_hw = (32, 32)
    # Boxes in stride-4 space: [y1, x1, y2, x2]
    boxes = torch.tensor([
        [0, 0, 8, 8],
        [0, 0, 1, 1],
        [127, 127, 128, 128],
        [10, 20, 50, 70],
    ])
    mapped = _map_boxes_between_grids(boxes, src_hw, dst_hw)
    assert mapped.shape == boxes.shape
    # All boxes must be valid and non-empty (y2 > y1, x2 > x1)
    assert (mapped[:, 2] > mapped[:, 0]).all()
    assert (mapped[:, 3] > mapped[:, 1]).all()
    # Bounds must stay within [0, dst_h] and [0, dst_w]
    assert (mapped[:, 0] >= 0).all() and (mapped[:, 1] >= 0).all()
    assert (mapped[:, 2] <= dst_hw[0]).all() and (mapped[:, 3] <= dst_hw[1]).all()


def test_native_pooling_output_shape():
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False)
    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        native_scale_pooling=True,
        regional_feature_stats="mean",
    )
    p4 = torch.randn(2, 32, 64, 64)
    p8 = torch.randn(2, 32, 32, 32)
    p16 = torch.randn(2, 32, 16, 16)
    out = head._collect_region_features((p4, p8, p16), regions)
    m_total = regions.boxes.shape[0]
    assert out.shape == (2, m_total, 33)
    assert torch.isfinite(out).all()


def test_native_pooling_32px_matches_p4_reference():
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False)
    head_native = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        native_scale_pooling=True,
    )
    head_legacy = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        native_scale_pooling=False,
    )
    p4 = torch.randn(1, 32, 64, 64)
    p8 = torch.randn(1, 32, 32, 32)
    p16 = torch.randn(1, 32, 16, 16)

    out_native = head_native._collect_region_features((p4, p8, p16), regions)
    out_legacy = head_legacy._collect_region_features((p4, p8, p16), regions)

    mask_32 = regions.scale_id == 0
    # For 32px regions, source is native p4 in both cases: must match exactly
    torch.testing.assert_close(out_native[:, mask_32], out_legacy[:, mask_32])


def test_native_pooling_parameter_count_unchanged():
    cfg_legacy = RMRv3Config(pretrained=False, native_scale_pooling=False)
    cfg_native = RMRv3Config(pretrained=False, native_scale_pooling=True)
    m_legacy = RMRv3(cfg_legacy)
    m_native = RMRv3(cfg_native)

    p_legacy = sum(p.numel() for p in m_legacy.parameters() if p.requires_grad)
    p_native = sum(p.numel() for p in m_native.parameters() if p.requires_grad)
    assert p_legacy == 101763
    assert p_native == 101763


def test_mean_std_constant_feature_zero_std():
    feat = torch.ones(2, 16, 32, 32) * 5.0
    boxes = torch.tensor([[0, 0, 8, 8], [4, 4, 16, 16]])
    out = region_mean_std_features(feat, boxes, eps=1e-8)
    # Shape: [2, num_boxes, 32] -> mean (16) + std (16)
    assert out.shape == (2, 2, 32)
    mean = out[:, :, :16]
    std = out[:, :, 16:]
    torch.testing.assert_close(mean, torch.ones_like(mean) * 5.0)
    # std of constant field should be 0.0 (or ~sqrt(1e-8)=1e-4)
    assert (std < 1e-3).all()


def test_mean_std_shape():
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False)
    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        regional_feature_stats="mean_std",
    )
    p4 = torch.randn(2, 32, 64, 64)
    p8 = torch.randn(2, 32, 32, 32)
    p16 = torch.randn(2, 32, 16, 16)
    out = head._collect_region_features((p4, p8, p16), regions)
    m_total = regions.boxes.shape[0]
    # 32 mean + 32 std + 1 scale = 65
    assert out.shape == (2, m_total, 65)


def test_mean_std_finite_gradients():
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False)
    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        regional_feature_stats="mean_std",
    )
    p4 = torch.randn(1, 32, 64, 64, requires_grad=True)
    p8 = torch.randn(1, 32, 32, 32, requires_grad=True)
    p16 = torch.randn(1, 32, 16, 16, requires_grad=True)

    out = head._collect_region_features((p4, p8, p16), regions)
    h = head.trunk(out)
    pred_mean = head.mean_head(h)
    loss = pred_mean.sum()
    loss.backward()

    assert p4.grad is not None and torch.isfinite(p4.grad).all()
    assert p8.grad is not None and torch.isfinite(p8.grad).all()
    assert p16.grad is not None and torch.isfinite(p16.grad).all()


def test_mean_std_model_under_105k():
    cfg = RMRv3Config(
        pretrained=False,
        regional_feature_stats="mean_std",
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Expected: 101,763 + (65 - 33)*48 = 101,763 + 1,536 = 103,299
    assert total_params == 103299
    assert total_params < 105000


def test_hierarchical_dm_block_loss_reduction():
    pred = torch.rand(2, 64, 64) * 5.0
    target = torch.randint(0, 5, (2, 64, 64)).float()

    l16 = flat_dm_block_loss(pred, target, block_px=16, stride=4)
    l32 = flat_dm_block_loss(pred, target, block_px=32, stride=4)
    l64 = flat_dm_block_loss(pred, target, block_px=64, stride=4)

    assert torch.isfinite(l16) and l16 > 0
    assert torch.isfinite(l32) and l32 > 0
    assert torch.isfinite(l64) and l64 > 0

    # Test backward-compatible flat_dm16_loss matches flat_dm_block_loss(16) exactly
    l16_compat = flat_dm16_loss(pred, target, stride=4)
    torch.testing.assert_close(l16, l16_compat)


def test_hierarchical_dm_loss_gradient_flow():
    pred = torch.rand(1, 64, 64, requires_grad=True)
    target = torch.randint(0, 5, (1, 64, 64)).float()

    loss = hierarchical_dm_loss(
        pred,
        target,
        block_sizes_px=(16, 32, 64),
        weights=(0.50, 0.30, 0.20),
        kappas=(20.0, 20.0, 20.0),
        stride=4,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_v3b_exact_backward_compatibility():
    # Verify that default RMRv3 config produces identical parameter counts and valid forward
    torch.manual_seed(42)
    cfg_default = RMRv3Config(pretrained=False)
    m = RMRv3(cfg_default)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 101763

    x = torch.randn(1, 3, 256, 256)
    out = m(x)
    assert "y" in out and "y0" in out and "iterates" in out
    assert len(out["iterates"]) == 3
    assert out["y"].shape == (1, 1, 64, 64)
