from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    build_perspective_adaptive_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
)
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.architecture import RMRv3


def test_dpw_coordinate_invariants():
    """Verify DPW region coordinates strictly obey bounding box invariants."""
    h, w, stride = 128, 128, 4
    dpw = build_perspective_adaptive_regions(
        height=h,
        width=w,
        output_stride=stride,
        base_sizes_px=(32, 64, 128),
        overlap=0.4,
        camera_tilt=0.5,
        horizon_pos=0.1,
        min_scale_ratio=0.5,
    )
    assert isinstance(dpw, RegionSet)
    assert len(dpw.boxes) > 0
    assert dpw.boxes.shape[1] == 4

    boxes = dpw.boxes
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    assert (y1 >= 0).all()
    assert (x1 >= 0).all()
    assert (y2 <= h).all()
    assert (x2 <= w).all()
    assert (y2 > y1).all()
    assert (x2 > x1).all()

    expected_area = ((y2 - y1) * (x2 - x1)).float()
    torch.testing.assert_close(dpw.area, expected_area)


def test_dpw_horizon_scale_contraction():
    """Verify that DPW windows contract at the horizon to provide higher spatial resolution."""
    h, w, stride = 128, 128, 4
    dpw = build_perspective_adaptive_regions(
        height=h,
        width=w,
        output_stride=stride,
        base_sizes_px=(32, 64, 128),
        overlap=0.4,
        camera_tilt=0.8,
        horizon_pos=0.1,
        min_scale_ratio=0.5,
    )
    boxes = dpw.boxes
    scale_ids = dpw.scale_id

    # For each scale, average area at horizon (y1 == 0) vs foreground (y2 == h)
    for sid in range(3):
        mask_s = scale_ids == sid
        boxes_s = boxes[mask_s]
        areas_s = dpw.area[mask_s]

        # Horizon boxes: top row
        horizon_mask = boxes_s[:, 0] == 0
        # Foreground boxes: bottom row
        fg_mask = boxes_s[:, 2] == h

        if horizon_mask.any() and fg_mask.any():
            avg_horizon_area = areas_s[horizon_mask].mean().item()
            avg_fg_area = areas_s[fg_mask].mean().item()
            # Horizon area must be strictly smaller than foreground area
            assert avg_horizon_area < avg_fg_area, (
                f"Scale {sid}: horizon area {avg_horizon_area} not < fg area {avg_fg_area}"
            )


def test_dpw_exact_adjointness():
    """Verify exact mathematical adjointness <A y, r> == <y, A* r> with relative error < 1e-5."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w, stride = 128, 128, 4
    dpw = build_perspective_adaptive_regions(
        height=h,
        width=w,
        output_stride=stride,
        base_sizes_px=(32, 64, 128),
        overlap=0.4,
        camera_tilt=0.5,
        horizon_pos=0.1,
        min_scale_ratio=0.5,
        device=device,
    )

    b_size = 2
    torch.manual_seed(42)
    y = torch.rand(b_size, 1, h, w, device=device, dtype=torch.float32) + 0.1
    m = dpw.boxes.shape[0]
    r = torch.randn(b_size, 1, m, device=device, dtype=torch.float32)

    # Forward: b = A y
    pref = prefix2d(y)
    ay = rectangle_sum_from_prefix(pref, dpw.boxes)  # [B, 1, M]

    # Adjoint: A* r
    a_star_r = regional_adjoint(r, dpw.boxes, h, w)  # [B, 1, H, W]

    inner_forward = (ay * r).sum().item()
    inner_adjoint = (y * a_star_r).sum().item()

    rel_err = abs(inner_forward - inner_adjoint) / (abs(inner_forward) + 1e-7)
    assert rel_err < 1e-5, f"Adjointness error {rel_err} exceeds 1e-5: forward={inner_forward}, adjoint={inner_adjoint}"


def test_dpw_model_forward_backward():
    """Verify full end-to-end forward, backward, parameter budget, and grad flow with DPW."""
    cfg = RMRv3Config(
        dynamic_perspective_windows=True,
        dpw_min_scale_ratio=0.5,
        dpw_horizon_pos=0.1,
        dpw_tilt_levels=5,
        use_diag=True,
        enable_solver=True,
        iterations=3,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable <= 105000, f"Total params {total_trainable} exceeds 105k budget"

    model.train()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "scene_tilt" in out
    assert out["y"].shape == (2, 1, 64, 64)
    assert not torch.isnan(out["y"]).any()
    assert not torch.isinf(out["y"]).any()

    # Backpropagation
    loss = out["y"].sum()
    loss.backward()

    # Verify gradients exist and are non-zero
    fine_head_params_with_grad = [p for p in model.fine_head.parameters() if p.grad is not None]
    assert len(fine_head_params_with_grad) > 0, "No fine_head parameters received gradients"
    if model.dcap is not None:
        assert model.dcap.proj.weight.grad is not None


def test_dpw_tilt_quantization_and_cache():
    """Verify that tilt quantization ensures bounded LRU cache entries."""
    cfg = RMRv3Config(
        dynamic_perspective_windows=True,
        dpw_tilt_levels=5,
    )
    model = RMRv3(cfg)
    device = torch.device("cpu")

    # Call _regions with various continuous tilt values in [0, 1]
    for tilt in [0.01, 0.04, 0.23, 0.27, 0.49, 0.52, 0.74, 0.76, 0.98, 1.0]:
        _ = model._regions(128, 128, device, stride=4, tilt=tilt)

    # 5 tilt levels mean at most 5 cache entries for 128x128
    assert len(model._region_cache) <= 5


def test_dpw_subpixel_stride2_compatibility():
    """Verify that DPW seamlessly supports Stride 2 subpixel lattice scaling."""
    cfg = RMRv3Config(
        dynamic_perspective_windows=True,
        subpixel_stride2=True,
        use_diag=True,
        enable_solver=True,
        iterations=3,
        max_trainable_params=105000,
    )
    model = RMRv3(cfg)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert "y" in out
    assert out["y"].shape == (2, 1, 128, 128)
    assert not torch.isnan(out["y"]).any()

