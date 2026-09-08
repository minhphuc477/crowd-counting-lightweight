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
    _map_boxes_to_stride,
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


def test_mean_std_fp32_numerical_precision_under_fp16():
    """Verify that region_mean_std_features handles FP16 inputs without catastrophic cancellation."""
    # Construct a high-mean, low-variance FP16 tensor
    feat_f16 = (torch.randn(2, 32, 64, 64) * 0.01 + 100.0).half()
    boxes = torch.tensor([[0, 0, 8, 8], [10, 10, 20, 20], [0, 0, 32, 32]])
    out = region_mean_std_features(feat_f16, boxes, eps=1e-6)

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    # Check that std values are strictly positive and plausible
    std = out[:, :, 32:]
    assert (std > 0).all()
    assert (std < 1.0).all()  # Std of noise was ~0.01


@pytest.mark.parametrize(
    "src_hw, dst_hw",
    [
        ((158, 212), (40, 53)),  # Downsampled from 631x847 odd SHA-A image
        ((129, 193), (33, 49)),  # Downsampled from 513x769
        ((125, 126), (32, 32)),  # Downsampled from 499x501
        ((64, 91), (16, 23)),    # P4 -> P16 odd stride mapping
    ],
)
def test_native_box_mapping_odd_dimensions(src_hw, dst_hw):
    """Test geometry preservation and lattice bounds for arbitrary non-divisible/odd feature shapes."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw

    # Generate synthetic candidate boxes spanning corners, edges, small, and large extents
    boxes = torch.tensor([
        [0, 0, 8, 8],
        [0, 0, 1, 1],
        [src_h - 8, src_w - 8, src_h, src_w],
        [src_h - 1, src_w - 1, src_h, src_w],
        [0, 0, src_h, src_w],
        [src_h // 4, src_w // 4, src_h // 2, src_w // 2],
    ])

    mapped = _map_boxes_between_grids(boxes, src_hw, dst_hw)

    # 1. Non-empty bounding boxes
    assert (mapped[:, 2] > mapped[:, 0]).all(), "All boxes must satisfy y2 > y1"
    assert (mapped[:, 3] > mapped[:, 1]).all(), "All boxes must satisfy x2 > x1"

    # 2. Strict lattice boundaries
    assert (mapped[:, 0] >= 0).all() and (mapped[:, 1] >= 0).all()
    assert (mapped[:, 2] <= dst_h).all() and (mapped[:, 3] <= dst_w).all()


def test_native_pooling_full_image_odd_forward_pass():
    """Verify end-to-end forward pass on an odd full-image dimension with native pooling and mean_std."""
    cfg = RMRv3Config(
        pretrained=False,
        native_scale_pooling=True,
        regional_feature_stats="mean_std",
    )
    model = RMRv3(cfg)
    model.eval()

    # 631x847 is an odd non-square image typical of SHA-A test set
    x = torch.randn(1, 3, 631, 847)
    with torch.no_grad():
        out = model(x)

    assert "y" in out and "y0" in out
    assert torch.isfinite(out["y"]).all()
    assert torch.isfinite(out["y0"]).all()
    assert torch.isfinite(out["b_region"]).all()
    assert torch.isfinite(out["region_dispersion"]).all()


def test_multiscale_dm_components_and_alias():
    """Verify multiscale_dm_loss returns correct granular components and hierarchical alias matches."""
    from rmr_v2.losses import multiscale_dm_loss

    pred = torch.rand(2, 64, 64) * 3.0
    target = torch.randint(0, 4, (2, 64, 64)).float()

    total, comps = multiscale_dm_loss(
        pred,
        target,
        block_sizes_px=(16, 32, 64),
        weights=(0.5, 0.3, 0.2),
        kappas=(20.0, 20.0, 20.0),
        stride=4,
        return_components=True,
    )

    assert set(comps.keys()) == {16, 32, 64}
    expected_total = 0.5 * comps[16] + 0.3 * comps[32] + 0.2 * comps[64]
    torch.testing.assert_close(total, expected_total)

    # Test alias produces exact same result
    alias_total = hierarchical_dm_loss(
        pred,
        target,
        block_sizes_px=(16, 32, 64),
        weights=(0.5, 0.3, 0.2),
        kappas=(20.0, 20.0, 20.0),
        stride=4,
    )
    torch.testing.assert_close(total, alias_total)


def test_strict_dm_config_guards():
    """Verify validate_v3_config rejects invalid DM configurations."""
    base_cfg = {
        "seed": 42,
        "model": {"output_stride": 4},
        "loss": {"use_multiscale_dm": True},
    }

    # 1. Empty block sizes
    cfg_empty = {**base_cfg, "loss": {"use_multiscale_dm": True, "dm_block_sizes_px": []}}
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_v3_config(cfg_empty)

    # 2. Block size non-positive
    cfg_neg_b = {**base_cfg, "loss": {"use_multiscale_dm": True, "dm_block_sizes_px": [0, 32], "dm_weights": [0.5, 0.5], "dm_kappas": [20.0, 20.0]}}
    with pytest.raises(ValueError, match="must be positive integers"):
        validate_v3_config(cfg_neg_b)

    # 3. Block size not divisible by output_stride (4)
    cfg_indivisible = {**base_cfg, "loss": {"use_multiscale_dm": True, "dm_block_sizes_px": [18, 32], "dm_weights": [0.5, 0.5], "dm_kappas": [20.0, 20.0]}}
    with pytest.raises(ValueError, match="must be divisible by model output_stride"):
        validate_v3_config(cfg_indivisible)

    # 4. Kappa non-positive
    cfg_neg_k = {**base_cfg, "loss": {"use_multiscale_dm": True, "dm_block_sizes_px": [16, 32], "dm_weights": [0.5, 0.5], "dm_kappas": [0.0, 20.0]}}
    with pytest.raises(ValueError, match="must be > 0"):
        validate_v3_config(cfg_neg_k)

    # 5. Weight sum <= 0
    cfg_zero_w = {**base_cfg, "loss": {"use_multiscale_dm": True, "dm_block_sizes_px": [16, 32], "dm_weights": [0.0, 0.0], "dm_kappas": [20.0, 20.0]}}
    with pytest.raises(ValueError, match="must sum to > 0"):
        validate_v3_config(cfg_zero_w)


def test_physical_stride_mapping_stability():
    """Verify physical stride mapping preserves constant cell extent across odd image grids.

    A 64px region (16 cells on P4, stride 4) mapped to P8 (stride 8) must span exactly 8 cells.
    A 128px region (32 cells on P4, stride 4) mapped to P16 (stride 16) must span exactly 8 cells.
    """
    # Case 1: 64px region on P4 (stride 4) -> 16 cells in width and height
    boxes_64 = torch.tensor([
        [0, 0, 16, 16],
        [8, 8, 24, 24],
        [16, 32, 32, 48],
    ])
    mapped_p8 = _map_boxes_to_stride(boxes_64, src_stride=4, dst_stride=8, dst_hw=(79, 106))
    h_span_p8 = mapped_p8[:, 2] - mapped_p8[:, 0]
    w_span_p8 = mapped_p8[:, 3] - mapped_p8[:, 1]
    assert (h_span_p8 == 8).all(), f"Expected 8 cells on P8 for 64px region, got {h_span_p8}"
    assert (w_span_p8 == 8).all(), f"Expected 8 cells on P8 for 64px region, got {w_span_p8}"

    # Case 2: 128px region on P4 (stride 4) -> 32 cells in width and height
    boxes_128 = torch.tensor([
        [0, 0, 32, 32],
        [16, 16, 48, 48],
    ])
    mapped_p16 = _map_boxes_to_stride(boxes_128, src_stride=4, dst_stride=16, dst_hw=(40, 53))
    h_span_p16 = mapped_p16[:, 2] - mapped_p16[:, 0]
    w_span_p16 = mapped_p16[:, 3] - mapped_p16[:, 1]
    assert (h_span_p16 == 8).all(), f"Expected 8 cells on P16 for 128px region, got {h_span_p16}"
    assert (w_span_p16 == 8).all(), f"Expected 8 cells on P16 for 128px region, got {w_span_p16}"


def test_train_config_bounds_guards():
    """Verify validate_v3_config rejects invalid train parameter bounds."""
    base_cfg = {"seed": 42, "model": {"output_stride": 4}}

    # lr <= 0
    with pytest.raises(ValueError, match="train.lr must be strictly positive"):
        validate_v3_config({**base_cfg, "train": {"lr": 0.0}})
    with pytest.raises(ValueError, match="train.lr must be strictly positive"):
        validate_v3_config({**base_cfg, "train": {"lr": -0.001}})

    # epochs < 1
    with pytest.raises(ValueError, match="train.epochs must be >= 1"):
        validate_v3_config({**base_cfg, "train": {"epochs": 0}})

    # eval_every < 1
    with pytest.raises(ValueError, match="train.eval_every must be >= 1"):
        validate_v3_config({**base_cfg, "train": {"eval_every": 0}})

    # batch_size < 1
    with pytest.raises(ValueError, match="train.batch_size must be >= 1"):
        validate_v3_config({**base_cfg, "train": {"batch_size": 0}})

    # grad_clip <= 0
    with pytest.raises(ValueError, match="train.grad_clip must be strictly positive"):
        validate_v3_config({**base_cfg, "train": {"grad_clip": 0.0}})

    # patience < 0
    with pytest.raises(ValueError, match="train.patience must be >= 0"):
        validate_v3_config({**base_cfg, "train": {"patience": -1}})
