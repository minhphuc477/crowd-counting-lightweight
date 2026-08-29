import numpy as np
import pytest
import torch
from PIL import Image

from hpc.data.common import BaseCrowdDataset
from hpc.data.transforms import NTPCGeometricTransform


def test_flip_twice_is_identity():
    """Applying horizontal flip twice must exactly restore the original point coordinates."""
    crop_size = 256
    pts = np.array([
        [0.0, 0.0],
        [10.5, 20.3],
        [127.5, 127.5],
        [255.0, 255.0],
    ], dtype=np.float32)

    # Flip once: x' = (crop_size - 1.0) - x
    pts_flipped = pts.copy()
    pts_flipped[:, 0] = (float(crop_size - 1.0)) - pts_flipped[:, 0]

    # Flip twice:
    pts_restored = pts_flipped.copy()
    pts_restored[:, 0] = (float(crop_size - 1.0)) - pts_restored[:, 0]

    np.testing.assert_allclose(pts, pts_restored, atol=1e-6)


def test_transform_bounds_invariant():
    """NTPCGeometricTransform must always produce points strictly within valid bounds."""
    np.random.seed(42)
    crop_size = 256
    transform = NTPCGeometricTransform(crop_size=crop_size, scale_range=(0.5, 2.0), flip_prob=0.5)

    for _ in range(20):
        w, h = np.random.randint(300, 800), np.random.randint(300, 800)
        img = Image.new("RGB", (w, h), color="gray")
        n_pts = np.random.randint(10, 100)
        pts = np.column_stack([
            np.random.uniform(0.0, float(w - 1.0), n_pts),
            np.random.uniform(0.0, float(h - 1.0), n_pts),
        ]).astype(np.float32)

        crop_img, crop_pts = transform(img, pts)
        assert crop_img.size == (crop_size, crop_size)

        if len(crop_pts) > 0:
            assert np.all(crop_pts[:, 0] >= 0.0)
            assert np.all(crop_pts[:, 0] <= float(crop_size - 1.0))
            assert np.all(crop_pts[:, 1] >= 0.0)
            assert np.all(crop_pts[:, 1] <= float(crop_size - 1.0))


def test_dataset_sample_point_tree_match(tmp_path):
    """BaseCrowdDataset must produce exact match between len(crop_pts) and gt_count."""
    img_path = str(tmp_path / "img_001.jpg")
    Image.new("RGB", (400, 400), color="white").save(img_path)
    points = np.array([[50.0, 50.0], [100.0, 100.0], [200.0, 200.0]], dtype=np.float32)

    ds = BaseCrowdDataset(
        image_paths=[img_path],
        points_list=[points],
        crop_size=256,
        is_train=True,
    )

    sample = ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert int(sample["gt_count"].item()) == len(sample["gt_points"])
