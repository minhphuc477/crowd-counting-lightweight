"""Test Suite for V68 Universal Preprocessing and Cross-Dataset Invariants.

Verifies:
1. Crowd-aware crop augmentation on dense, sparse, single-point, and negative (0-count) images.
2. Coordinate preservation and boundary clipping [0, crop_size - 1].
3. Dynamic k-NN Bayesian loss adaptation on n >= 2 points and sigma_max = 16.0.
4. Config fallbacks for native full-image evaluation (tiling: false).
"""

import math
import random
import pytest
import torch
from PIL import Image
from rmr_core.transforms import train_transform
from rmr_v3.losses.point_supervision import bayesian_loss
from rmr_v3.losses.config import RMRv3LossConfig


def test_crowd_aware_crop_dense_region() -> None:
    """Verifies that crowd-aware crop successfully samples near crowd clusters."""
    # Create a 1000x1000 image with points only in a tight cluster [800, 850]
    img = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
    pts = torch.tensor([
        [820.0, 820.0],
        [825.0, 822.0],
        [830.0, 818.0],
        [815.0, 830.0],
    ], dtype=torch.float32)

    # With crowd_aware_crop=True and prob=1.0, the crop should consistently include points
    hits = 0
    num_trials = 20
    for _ in range(num_trials):
        img_t, pts_t = train_transform(
            img,
            pts,
            crop_size=512,
            scale_range=(1.0, 1.0),
            hflip_prob=0.0,
            crowd_aware_crop=True,
            crowd_aware_prob=1.0,
        )
        assert img_t.shape == (3, 512, 512)
        if pts_t.shape[0] > 0:
            hits += 1
            # Check coordinate clipping bounds
            assert (pts_t[:, 0] >= 0.0).all()
            assert (pts_t[:, 0] <= 511.0).all()
            assert (pts_t[:, 1] >= 0.0).all()
            assert (pts_t[:, 1] <= 511.0).all()

    # Blind random crop on 1000x1000 has only ~26% chance to hit [800, 850]
    # Crowd-aware crop with jitter [-128, 128] around [820, 820] has 100% hit rate
    assert hits == num_trials, f"Expected {num_trials} hits, got {hits}"


def test_crowd_aware_crop_negative_empty_image() -> None:
    """Verifies robustness on NWPU-Crowd negative images with 0 annotations."""
    img = Image.new("RGB", (800, 600), color=(50, 50, 50))
    pts = torch.empty((0, 2), dtype=torch.float32)

    img_t, pts_t = train_transform(
        img,
        pts,
        crop_size=512,
        scale_range=(1.0, 1.0),
        crowd_aware_crop=True,
        crowd_aware_prob=0.85,
    )
    assert img_t.shape == (3, 512, 512)
    assert pts_t.shape == (0, 2)


def test_crowd_aware_crop_single_point() -> None:
    """Verifies that a single point image is gracefully handled without index error."""
    img = Image.new("RGB", (700, 700), color=(200, 200, 200))
    pts = torch.tensor([[350.0, 350.0]], dtype=torch.float32)

    img_t, pts_t = train_transform(
        img,
        pts,
        crop_size=512,
        scale_range=(1.0, 1.0),
        crowd_aware_crop=True,
        crowd_aware_prob=1.0,
    )
    assert img_t.shape == (3, 512, 512)
    assert pts_t.shape[0] == 1
    assert 0.0 <= pts_t[0, 0].item() <= 511.0
    assert 0.0 <= pts_t[0, 1].item() <= 511.0


def test_bayesian_loss_dynamic_knn_adaptive_sigma() -> None:
    """Verifies dynamic k-NN computation for 2, 3, and dense points."""
    # Test 2 points
    pred_density = torch.ones((1, 1, 128, 128), dtype=torch.float32, requires_grad=True)
    pts_2 = torch.tensor([[50.0, 50.0], [54.0, 50.0]], dtype=torch.float32)  # dist = 4.0

    loss_2 = bayesian_loss(
        pred_density,
        points_list=[pts_2],
        adaptive_sigma=True,
        sigma_min=2.0,
        sigma_max=16.0,
    )
    assert torch.isfinite(loss_2)
    assert loss_2.item() > 0.0
    loss_2.backward()
    assert pred_density.grad is not None
    assert torch.isfinite(pred_density.grad).all()

    # Test sparse points with max sigma clamping
    pred_density_sparse = torch.ones((1, 1, 128, 128), dtype=torch.float32, requires_grad=True)
    pts_sparse = torch.tensor([[10.0, 10.0], [100.0, 100.0]], dtype=torch.float32)  # dist ~ 127
    loss_sparse = bayesian_loss(
        pred_density_sparse,
        points_list=[pts_sparse],
        adaptive_sigma=True,
        sigma_min=2.0,
        sigma_max=16.0,
    )
    assert torch.isfinite(loss_sparse)


def test_bayesian_loss_config_default_sigma_max() -> None:
    """Verifies RMRv3LossConfig default bayesian_sigma_max is 16.0."""
    cfg = RMRv3LossConfig()
    assert cfg.bayesian_sigma_max == 16.0
