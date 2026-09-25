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

    # Parameter count: Linear(32, 1 + 3) -> 32 * 4 + 4 = 132 parameters
    num_params = sum(p.numel() for p in dcap.parameters())
    assert num_params == 132, f"Expected 132 params for DCAP, got {num_params}"

    # Step 0 Identity Parity: With zero-initialized weights, outputs neutral state
    p16 = torch.randn(2, 32, 32, 32)
    scene_tilt, delta_scale = dcap(p16)

    assert scene_tilt.shape == (2, 1, 1, 1)
    assert delta_scale.shape == (2, 3, 1, 1)

    # scene_tilt should be sigmoid(0) = 0.5
    assert torch.allclose(scene_tilt, torch.full_like(scene_tilt, 0.5), atol=1e-5)
    # delta_scale should be 0.0 identically
    assert torch.allclose(delta_scale, torch.zeros_like(delta_scale), atol=1e-5)


def test_diag_scale_routing_translation_equivariance_and_partition() -> None:
    """Verify DiAG preserves exact translation equivariance and partition of unity."""
    in_channels = 32
    num_scales = 3
    router = DiAGScaleRoutingHead(in_channels=in_channels, num_scales=num_scales)

    b, c, h, w = 1, 32, 64, 64
    x = torch.randn(b, c, h, w)

    # 1. Step 0 uniform partition
    pi = router(x)
    assert pi.shape == (b, num_scales, h, w)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / num_scales), atol=1e-5)
    assert torch.allclose(pi.sum(dim=1), torch.ones(b, h, w), atol=1e-5)

    # 2. Perturb router weights to non-zero
    with torch.no_grad():
        router.pw.weight.normal_(std=0.1)
        router.pw.bias.normal_(std=0.1)

    pi_pert = router(x)
    # Partition of unity must hold identically regardless of weights
    assert torch.allclose(pi_pert.sum(dim=1), torch.ones(b, h, w), atol=1e-5)

    # 3. Translation Equivariance: Pure dynamic convolution preserves translation equivariance
    x_supp = torch.zeros(1, 32, 64, 64)
    x_supp[:, :, 16:48, 16:48] = torch.randn(1, 32, 32, 32)
    shift_y, shift_x = 2, 3
    x_supp_shifted = torch.roll(x_supp, shifts=(shift_y, shift_x), dims=(-2, -1))
    pi_supp = router(x_supp)
    pi_supp_shifted = router(x_supp_shifted)
    expected_shifted = torch.roll(pi_supp, shifts=(shift_y, shift_x), dims=(-2, -1))
    assert torch.allclose(pi_supp_shifted, expected_shifted, atol=1e-5), (
        "Pure dynamic router must be strictly translation-equivariant!"
    )


def test_rmr_v34_total_trainable_parameters() -> None:
    """Verify that RMR-v34 with DiAG satisfies strict parameter budget <= 105,000."""
    cfg = RMRv3Config(
        use_diag=True,
        dynamic_scale_routing=False,
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
    """Verify complete forward pass with DiAG outputs scene_tilt and delta_scale."""
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
    assert "scene_tilt" in out
    assert "delta_scale" in out

    assert out["scene_tilt"].shape == (2, 1, 1, 1)
    assert out["delta_scale"].shape == (2, 3, 1, 1)
    assert (out["scene_tilt"] >= 0.0).all() and (out["scene_tilt"] <= 1.0).all()


def test_diag_dcap_active_gradient_flow() -> None:
    """Verify that both scene_tilt (row 0) and delta_scale (rows 1-3) receive active gradients."""
    dcap = DynamicCameraAnglePredictor(in_channels=32, num_scales=3)
    router = DiAGScaleRoutingHead(in_channels=32, num_scales=3)

    # Initialize router with non-zero weights so logits != 0
    with torch.no_grad():
        router.pw.weight.normal_(std=0.1)

    p16 = torch.randn(2, 32, 16, 16, requires_grad=True)
    p4 = torch.randn(2, 32, 64, 64, requires_grad=True)

    scene_tilt, delta_scale = dcap(p16)
    pi = router(p4, delta_scale=delta_scale, scene_tilt=scene_tilt)

    loss = (pi[:, 0] * 2.0).sum()
    loss.backward()

    assert dcap.proj.weight.grad is not None
    assert torch.isfinite(dcap.proj.weight.grad).all()

    # Row 0 is scene_tilt; rows 1-3 are delta_scale
    grad_tilt = dcap.proj.weight.grad[0].abs().sum().item()
    grad_delta = dcap.proj.weight.grad[1:].abs().sum().item()

    assert grad_tilt > 0.0, f"scene_tilt received zero gradient: {grad_tilt}"
    assert grad_delta > 0.0, f"delta_scale received zero gradient: {grad_delta}"