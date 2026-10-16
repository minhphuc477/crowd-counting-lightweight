from __future__ import annotations

import pytest
import torch

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
    regional_sum,
)
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.architecture import RMRv3


def test_dpw_permanently_banned():
    """Verify that dynamic_perspective_windows=True is strictly banned with a clear error."""
    with pytest.raises(ValueError, match="dynamic_perspective_windows=True is permanently BANNED"):
        RMRv3Config(dynamic_perspective_windows=True)


def test_canonical_multiscale_regions_invariants():
    """Verify that canonical multiscale regions maintain 100% mathematical integrity."""
    h, w, stride = 128, 128, 4
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=stride,
        region_sizes_px=(32, 64, 128),
        overlap=0.5,
        include_full_image=False,
    )
    assert isinstance(regions, RegionSet)
    assert len(regions.boxes) == 1235
    assert regions.boxes.shape[1] == 4

    boxes = regions.boxes
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    assert (y1 >= 0).all()
    assert (x1 >= 0).all()
    assert (y2 <= h).all()
    assert (x2 <= w).all()
    assert (y2 > y1).all()
    assert (x2 > x1).all()

    expected_area = ((y2 - y1) * (x2 - x1)).float()
    torch.testing.assert_close(regions.area, expected_area)


def test_canonical_exact_adjointness():
    """Verify exact adjointness <Ay, r> == <y, A*r> on universal multiscale dictionary."""
    torch.manual_seed(42)
    b, c, h, w = 2, 1, 64, 64
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32, 64, 128),
        overlap=0.5,
        include_full_image=False,
    )

    y = torch.randn(b, c, h, w, dtype=torch.float32)
    ay = regional_sum(y, regions.boxes)  # [B, 1, M]
    r = torch.randn_like(ay)  # [B, 1, M]

    astar_r = regional_adjoint(r, regions.boxes, h, w)  # [B, 1, H, W]

    lhs = (ay * r).sum()
    rhs = (y * astar_r).sum()

    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_model_forward_backward_with_canonical_regions():
    """Verify end-to-end forward and backward passes with canonical multiscale dictionary."""
    torch.manual_seed(42)
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        region_sizes_px=(32, 64, 128),
        region_overlap=0.5,
        enable_solver=True,
        iterations=2,
        use_diag=True,
    )
    model = RMRv3(cfg)
    model.train()

    x = torch.randn(1, 3, 128, 128, requires_grad=True)
    out = model(x)

    assert out.y is not None
    assert out.y0 is not None
    assert out.b_solver is not None
    assert out.y.shape == (1, 1, 32, 32)

    loss = out.y.sum() + out.y0.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
