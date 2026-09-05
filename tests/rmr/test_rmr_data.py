import torch

from rmr_count.data import rasterize_points


def test_rasterize_points_conserves_count():
    pts = torch.tensor([[0.0, 0.0], [3.6, 4.0], [7.4, 7.4], [15.0, 15.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 4


def test_oob_points_are_ignored_not_clipped():
    pts = torch.tensor([[-1.0, 2.0], [2.0, 2.0], [20.0, 3.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 1


def test_rasterize_points_conserves_edge_points():
    # Boundary conservation: point at x=15.9, y=15.9 in 16x16 image with stride 4.
    # floor((15.9+0.5)/4) = 4 >= gw=4.
    # The point is valid (inside image) and must land in the last cell (j=3, i=3),
    # not be silently dropped.
    pts = torch.tensor([[15.9, 15.9], [0.0, 0.0], [15.1, 0.5]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 3
    assert y[0, 3, 3].item() == 1
    assert y[0, 0, 0].item() == 1
    assert y[0, 0, 3].item() == 1
