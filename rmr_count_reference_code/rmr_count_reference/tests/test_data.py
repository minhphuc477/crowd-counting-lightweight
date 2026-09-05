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
