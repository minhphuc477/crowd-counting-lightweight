from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.perspective_geometry import (
    DynamicCameraAnglePredictor,
    DiAGScaleRoutingHead,
)
from rmr_core.operators.regions import build_multiscale_regions, partition_regions_by_scale
from rmr_v3.losses.auxiliary import (
    count_invariant_cell_loss,
    topk_hard_background_loss,
    physical_scale_alignment_loss,
)


def test_dcap_parameters_and_step0_parity() -> None:
    """Verify DCAP parameter budget and Step 0 Identity Parity."""
    in_channels = 32
    num_scales = 3
    dcap = DynamicCameraAnglePredictor(in_channels=in_channels, num_scales=num_scales)

    # Parameter count: Linear(32, 2 + 3) -> 32 * 5 + 5 = 165 parameters
    num_params = sum(p.numel() for p in dcap.parameters())
    assert num_params == 165, f"Expected 165 params for DCAP, got {num_params}"

    # Step 0 Identity Parity: With zero-initialized weights, outputs neutral state
    p16 = torch.randn(2, 32, 32, 32)
    alpha, v_horizon, delta_scale = dcap(p16)

    assert alpha.shape == (2, 1, 1, 1)
    assert v_horizon.shape == (2, 1, 1, 1)
    assert delta_scale.shape == (2, 3, 1, 1)

    # alpha should be sigmoid(0) = 0.5
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5), atol=1e-5)
    # v_horizon should be 0.3 * tanh(0) = 0.0
    assert torch.allclose(v_horizon, torch.zeros_like(v_horizon), atol=1e-5)
    # delta_scale should be 0.0
    assert torch.allclose(delta_scale, torch.zeros_like(delta_scale), atol=1e-5)


def test_diag_scale_routing_overhead_invariance() -> None:
    """When alpha_persp = 0 (overhead drone/aerial view), vertical perspective gradient vanishes."""
    in_channels = 32
    num_scales = 3
    router = DiAGScaleRoutingHead(in_channels=in_channels, num_scales=num_scales, persp_slope_init="physical")

    b, c, h, w = 1, 32, 64, 64
    x = torch.zeros(b, c, h, w)  # zero carrier feature

    # Case 1: alpha_persp = 0.0 (Pure Overhead view)
    alpha_overhead = torch.zeros(b, 1, 1, 1)
    pi_overhead = router(x, alpha_persp=alpha_overhead)

    # The routing weights across all y scanlines must be perfectly identical (zero vertical bias)
    for y_idx in range(h):
        assert torch.allclose(pi_overhead[:, :, y_idx, :], pi_overhead[:, :, 0, :], atol=1e-5), (
            f"Overhead view (alpha=0) must be vertically invariant, but scanline {y_idx} differs from 0!"
        )

    # Case 2: alpha_persp = 1.0 (Steep oblique ground view)
    alpha_oblique = torch.ones(b, 1, 1, 1)
    pi_oblique = router(x, alpha_persp=alpha_oblique)

    # Near horizon (y=0, top): fine scale (k=0) must have higher probability than near foreground (y=H-1, bottom)
    prob_top_fine = pi_oblique[0, 0, 0, w // 2].item()
    prob_bottom_fine = pi_oblique[0, 0, h - 1, w // 2].item()
    assert prob_top_fine > prob_bottom_fine, (
        f"In oblique view, top (horizon) fine scale prob ({prob_top_fine:.3f}) "
        f"must be higher than bottom fine scale prob ({prob_bottom_fine:.3f})"
    )

    # Near foreground (y=H-1, bottom): coarse scale (k=2) must have higher probability than near horizon (y=0, top)
    prob_top_coarse = pi_oblique[0, 2, 0, w // 2].item()
    prob_bottom_coarse = pi_oblique[0, 2, h - 1, w // 2].item()
    assert prob_bottom_coarse > prob_top_coarse, (
        f"In oblique view, bottom coarse scale prob ({prob_bottom_coarse:.3f}) "
        f"must be higher than top coarse scale prob ({prob_top_coarse:.3f})"
    )


def test_rmr_v34_total_trainable_parameters() -> None:
    """Verify that RMR-v34 with DiAG satisfies strict parameter budget <= 105,000."""
    cfg = RMRv3Config(
        use_diag=True,
        dynamic_scale_routing=False,
        park_routing=False,
        hurdle_head=True,
        resonant_adjoint=True,
        resonant_adjoint_lambda=0.5,
        region_sizes_px=(32, 64, 128),
        region_head_hidden=48,
        regional_feature_stats="mean",
        iterations=6,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {total_params}")
    assert total_params <= 105000, f"Exceeded strict ceiling of 105,000: got {total_params}"
    # Verify headroom
    headroom = 105000 - total_params
    assert headroom >= 300, f"Expected at least 300 params headroom, got {headroom}"


def test_partition_parity_zero_clamping() -> None:
    """Verify that 3-scale universal dictionary partitions into 3 non-empty sets with 0 clamping."""
    regions = build_multiscale_regions(
        height=128, width=128, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5, include_full_image=False
    )
    partitions = partition_regions_by_scale(regions, k_scales=3)
    assert len(partitions) == 3

    # All 3 scales must be non-empty
    for k, mask_k, boxes_k in partitions:
        assert mask_k is not None
        assert boxes_k is not None
        assert boxes_k.shape[0] > 0, f"Scale {k} has 0 boxes!"

    # Total boxes must equal 1,235
    total_boxes = sum(b.shape[0] for _, _, b in partitions)
    assert total_boxes == 1235, f"Expected 1,235 total boxes, got {total_boxes}"


def test_dsmp_empty_background_suppression() -> None:
    """Verify that DSMP loss components penalize false background ripples without gradient starvation."""
    b, h, w = 2, 128, 128
    target_empty = torch.zeros(b, 1, h, w)  # pure background
    pred_with_ripple = torch.full((b, 1, h, w), 0.005)  # low-amplitude ripple

    # 1. Top-K Hard Background Mining Loss
    loss_hard_bg = topk_hard_background_loss(pred_with_ripple, target_empty, ratio=0.05)
    assert loss_hard_bg.item() > 0.0, "Hard background loss must strictly penalize background ripples"

    # 2. CI-Cell v2 Loss on empty background
    loss_ci_cell = count_invariant_cell_loss(pred_with_ripple, target_empty, alpha=2.0, beta=1.0)
    assert loss_ci_cell.item() > 0.0, "CI-Cell loss must strictly penalize background ripples"


def test_model_forward_diag_outputs() -> None:
    """Verify complete forward pass with DiAG outputs alpha_persp, v_horizon, delta_scale."""
    cfg = RMRv3Config(
        use_diag=True,
        region_sizes_px=(32, 64, 128),
        hurdle_head=True,
        resonant_adjoint=True,
        iterations=2,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "alpha_persp" in out
    assert "v_horizon" in out
    assert "delta_scale" in out

    assert out["alpha_persp"].shape == (2, 1, 1, 1)
    assert out["v_horizon"].shape == (2, 1, 1, 1)
    assert out["delta_scale"].shape == (2, 3, 1, 1)
    assert (out["alpha_persp"] >= 0.0).all() and (out["alpha_persp"] <= 1.0).all()
