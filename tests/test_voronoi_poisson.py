"""Unit tests for Scale-Normalized Voronoi-Poisson (SN-VP) Loss."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rmr_v3.losses.voronoi_poisson import (
    ScaleNormalizedVoronoiPoissonLoss,
    scale_normalized_voronoi_poisson_loss,
)


def test_forward_shape_and_scalar() -> None:
    """Verifies loss returns a single Float32 scalar tensor."""
    b, h, w = 2, 64, 64
    pred = torch.rand(b, 1, h, w, requires_grad=True) * 0.1
    pts = [
        torch.tensor([[10.0, 20.0], [50.0, 60.0]]),
        torch.tensor([[15.0, 30.0]]),
    ]
    loss = scale_normalized_voronoi_poisson_loss(pred, pts, stride=4)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert loss.item() >= 0.0


def test_backward_gradients() -> None:
    """Verifies loss backpropagates valid, non-NaN gradients to predicted density."""
    b, h, w = 2, 32, 32
    pred = (torch.rand(b, 1, h, w) * 0.05).requires_grad_(True)
    pts = [
        torch.tensor([[8.0, 8.0], [24.0, 24.0]]),
        torch.tensor([[16.0, 16.0]]),
    ]
    loss = scale_normalized_voronoi_poisson_loss(pred, pts, stride=4)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()
    assert pred.grad.abs().sum() > 0.0


def test_zero_crowd_background_vacuum() -> None:
    """Verifies pure background images (N=0) are penalized uniformly with gradient +1.0."""
    b, h, w = 1, 32, 32
    pred = (torch.ones(b, 1, h, w) * 0.02).requires_grad_(True)
    pts = [torch.empty(0, 2)]

    loss = scale_normalized_voronoi_poisson_loss(pred, pts, stride=4)
    expected_count = pred.sum()
    assert torch.allclose(loss, expected_count)

    loss.backward()
    assert pred.grad is not None
    assert torch.allclose(pred.grad, torch.ones_like(pred.grad))


def test_dense_crowd_zero_dilution() -> None:
    """Verifies adjacent heads (distance = 2px) receive unattenuated separate Voronoi supervision."""
    b, h, w = 1, 64, 64
    stride = 4
    img_size = (h * stride, w * stride)  # 256 x 256

    # Two heads separated by only 2 pixels: (100.0, 100.0) and (102.0, 100.0)
    pts = [torch.tensor([[100.0, 100.0], [102.0, 100.0]])]
    
    # Case A: Undercounting (total pred = 4096 * 0.0001 = 0.4096 < 2.0)
    pred_under = (torch.zeros(b, 1, h, w) + 0.0001).requires_grad_(True)
    loss_under = scale_normalized_voronoi_poisson_loss(
        pred_under, pts, stride=stride, img_size=img_size, lambda_center=0.0, lambda_global=0.0
    )
    loss_under.backward()

    assert pred_under.grad is not None
    assert not torch.isnan(pred_under.grad).any()
    # In both cells, predicted mass < 1.0, so gradient is strictly negative (pushing up)
    assert torch.allclose(pred_under.grad, torch.full_like(pred_under.grad, -1.0))

    # Case B: Overcounting (total pred = 4096 * 0.01 = 40.96 > 2.0)
    pred_over = (torch.zeros(b, 1, h, w) + 0.01).requires_grad_(True)
    loss_over = scale_normalized_voronoi_poisson_loss(
        pred_over, pts, stride=stride, img_size=img_size, lambda_center=0.0, lambda_global=0.0
    )
    loss_over.backward()

    assert pred_over.grad is not None
    assert not torch.isnan(pred_over.grad).any()
    # In both cells, predicted mass > 1.0, so gradient is strictly positive (pushing down)
    assert torch.allclose(pred_over.grad, torch.full_like(pred_over.grad, 1.0))


def test_module_wrapper() -> None:
    """Verifies ScaleNormalizedVoronoiPoissonLoss nn.Module produces identical outputs."""
    b, h, w = 2, 32, 32
    pred = torch.rand(b, 1, h, w) * 0.1
    pts = [
        torch.tensor([[10.0, 12.0]]),
        torch.tensor([[20.0, 22.0], [5.0, 5.0]]),
    ]
    criterion = ScaleNormalizedVoronoiPoissonLoss(stride=4, lambda_center=0.1, lambda_global=0.5)
    loss_module = criterion(pred, pts)
    loss_func = scale_normalized_voronoi_poisson_loss(
        pred, pts, stride=4, lambda_center=0.1, lambda_global=0.5
    )
    assert torch.allclose(loss_module, loss_func)


def test_device_cuda_and_amp() -> None:
    """Verifies GPU acceleration and AMP autocast stability if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    b, h, w = 2, 64, 64
    pred = (torch.rand(b, 1, h, w, device=device) * 0.1).requires_grad_(True)
    pts = [
        torch.tensor([[20.0, 30.0], [80.0, 90.0]], device=device),
        torch.tensor([[50.0, 50.0]], device=device),
    ]

    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = scale_normalized_voronoi_poisson_loss(pred, pts, stride=4)

    assert loss.dtype == torch.float32
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
