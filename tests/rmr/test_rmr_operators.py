import torch

from rmr_count.operators import (
    build_multiscale_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
    regional_sum,
)


def test_rectangle_sum_matches_naive():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 11, 13, dtype=torch.float64)
    boxes = torch.tensor([[0, 0, 3, 4], [2, 5, 11, 13], [7, 1, 10, 8]], dtype=torch.long)
    got = regional_sum(x, boxes)
    want = []
    for y1, x1, y2, x2 in boxes.tolist():
        want.append(x[..., y1:y2, x1:x2].sum(dim=(-2, -1)))
    want = torch.stack(want, dim=-1)
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_adjoint_identity():
    torch.manual_seed(1)
    b, c, h, w = 2, 2, 12, 15
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    x = torch.randn(b, c, h, w, dtype=torch.float64)
    e = torch.randn(b, c, regions.boxes.shape[0], dtype=torch.float64)
    ax = regional_sum(x, regions.boxes)
    ate = regional_adjoint(e, regions.boxes, h, w)
    lhs = (ax * e).sum()
    rhs = (x * ate).sum()
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_full_image_region_present():
    r = build_multiscale_regions(9, 10, output_stride=4, region_sizes_px=(16,), include_full_image=True)
    assert any(tuple(b.tolist()) == (0, 0, 9, 10) for b in r.boxes)
