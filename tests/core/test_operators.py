import pytest
import torch

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    prefix2d,
    rectangle_sum_from_prefix,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)


def test_prefix2d_and_rectangle_sum():
    x = torch.ones((2, 1, 32, 32), dtype=torch.float32)
    pref = prefix2d(x)
    assert pref.shape == (2, 1, 33, 33)

    boxes = torch.tensor([[0, 0, 8, 8], [4, 4, 12, 12], [0, 0, 32, 32]], dtype=torch.long)
    sums = rectangle_sum_from_prefix(pref, boxes)
    assert sums.shape == (2, 1, 3)
    assert torch.allclose(sums[0, 0, 0], torch.tensor(64.0))
    assert torch.allclose(sums[0, 0, 1], torch.tensor(64.0))
    assert torch.allclose(sums[0, 0, 2], torch.tensor(1024.0))


def test_regional_sum_and_adjoint_identity():
    torch.manual_seed(42)
    b, c, h, w = 2, 1, 32, 32
    x = torch.randn(b, c, h, w, dtype=torch.float32)
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)

    ax = regional_sum(x, regions.boxes)
    v = torch.randn_like(ax)
    at_v = regional_adjoint(v, regions.boxes, h, w)

    # <A x, v> == <x, A^T v>
    lhs = (ax * v).sum()
    rhs = (x * at_v).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_build_multiscale_regions_cache():
    r1 = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64), overlap=0.5)
    r2 = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64), overlap=0.5)
    assert r1.boxes.shape == r2.boxes.shape
    assert torch.equal(r1.boxes, r2.boxes)
    assert torch.equal(r1.scale_id, r2.scale_id)


def test_region_geometry():
    boxes = torch.tensor([[0, 0, 8, 8], [0, 0, 8, 16]], dtype=torch.long)
    geom = region_geometry(boxes, 32, 32)
    assert geom.shape == (2, 4)
    assert torch.all(torch.isfinite(geom))
