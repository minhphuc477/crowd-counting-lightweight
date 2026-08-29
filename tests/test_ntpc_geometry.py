import random
import numpy as np
import pytest
import torch
from PIL import Image

from hpc.data.common import BaseCrowdDataset, validate_point_annotations
from hpc.data.nwpu import resolve_nwpu_split_file
from hpc.data.point_counts import points_to_y4
from hpc.data.transforms import (
    NTPCGeometricTransform,
    compute_isotropic_resize,
    in_closed_pixel_support,
)


def test_flip_twice_is_identity():
    """Applying horizontal flip twice must exactly restore the original point coordinates."""
    crop_size = 256
    pts = np.array([
        [-0.5, -0.5],
        [0.0, 0.0],
        [10.5, 20.3],
        [127.5, 127.5],
        [255.0, 255.0],
        [255.49, 255.49],
    ], dtype=np.float32)

    # Flip once: x' = (crop_size - 1.0) - x
    pts_flipped = pts.copy()
    pts_flipped[:, 0] = (float(crop_size) - 1.0) - pts_flipped[:, 0]

    # Flip twice:
    pts_restored = pts_flipped.copy()
    pts_restored[:, 0] = (float(crop_size) - 1.0) - pts_restored[:, 0]

    np.testing.assert_allclose(pts, pts_restored, atol=1e-6)


def test_exact_boundary_flip_reflection_closed():
    """Points at exact boundaries -0.5 and W-0.5 must reflect to each other and remain within closed support."""
    crop_size = 256
    pts = np.array([
        [-0.5, 100.0],
        [float(crop_size) - 0.5, 100.0],
    ], dtype=np.float32)

    flipped_x = (float(crop_size) - 1.0) - pts[:, 0]

    assert flipped_x[0] == pytest.approx(float(crop_size) - 0.5)
    assert flipped_x[1] == pytest.approx(-0.5)
    assert np.all(flipped_x >= -0.5) and np.all(flipped_x <= float(crop_size) - 0.5)


def test_isotropic_scaling_preserves_aspect_ratio():
    """Transform must preserve aspect ratio for non-square aspect ratios (e.g. 100x1000, 300x1000)."""
    random.seed(42)
    np.random.seed(42)
    crop_size = 256

    test_shapes = [
        (100, 1000),
        (300, 1000),
        (800, 200),
        (1200, 400),
        (500, 500),
    ]

    for old_w, old_h in test_shapes:
        sampled_scale = random.uniform(0.7, 1.3)
        new_w, new_h = compute_isotropic_resize(old_w, old_h, crop_size, sampled_scale)

        sx = new_w / float(old_w)
        sy = new_h / float(old_h)

        # Aspect ratio distortion must be negligible (due only to 1px integer rounding)
        aspect_ratio_error = abs(sx - sy) / min(sx, sy)
        assert aspect_ratio_error < 0.02, (
            f"Aspect ratio distorted for ({old_w}, {old_h}): sx={sx:.4f}, sy={sy:.4f}, error={aspect_ratio_error:.4f}"
        )


def test_transform_bounds_invariant():
    """NTPCGeometricTransform must always produce points strictly within closed support [-0.5, crop-0.5]."""
    random.seed(42)
    np.random.seed(42)
    crop_size = 256
    transform = NTPCGeometricTransform(crop_size=crop_size, scale_range=(0.5, 2.0), flip_prob=0.5)

    for _ in range(25):
        w, h = np.random.randint(150, 800), np.random.randint(150, 800)
        img = Image.new("RGB", (w, h), color="gray")
        n_pts = np.random.randint(10, 100)
        pts = np.column_stack([
            np.random.uniform(0.0, float(w - 1.0), n_pts),
            np.random.uniform(0.0, float(h - 1.0), n_pts),
        ]).astype(np.float32)

        crop_img, crop_pts = transform(img, pts)
        assert crop_img.size == (crop_size, crop_size)

        if len(crop_pts) > 0:
            assert in_closed_pixel_support(crop_pts, crop_size, crop_size).all()


def test_sha_coordinate_base_conversion():
    """1-based MATLAB coordinates [1, W] x [1, H] must convert to 0-based [0, W-1] x [0, H-1]."""
    w, h = 400, 300
    mat_pts_1based = np.array([
        [1.0, 1.0],
        [float(w), float(h)],
        [200.0, 150.0],
    ], dtype=np.float32)

    converted = validate_point_annotations(mat_pts_1based, source="test", coordinate_base=1, image_shape=(w, h))

    expected = np.array([
        [0.0, 0.0],
        [float(w - 1.0), float(h - 1.0)],
        [199.0, 149.0],
    ], dtype=np.float32)

    np.testing.assert_allclose(converted, expected, atol=1e-6)
    assert np.all(converted[:, 0] >= 0.0) and np.all(converted[:, 0] <= float(w - 1.0))
    assert np.all(converted[:, 1] >= 0.0) and np.all(converted[:, 1] <= float(h - 1.0))


def test_out_of_bounds_annotation_fails_fast():
    """Annotations significantly beyond image bounds must raise ValueError, not silently clip."""
    w, h = 400, 300
    bad_pts = np.array([
        [100.0, 100.0],
        [w + 50.0, 100.0],  # Out of bounds
    ], dtype=np.float32)

    with pytest.raises(ValueError, match="Out-of-bounds annotation"):
        validate_point_annotations(bad_pts, source="test_bad", coordinate_base=0, image_shape=(w, h))


def test_shanghai_policy_preserves_raw_out_of_bounds_points_without_clipping():
    """The explicit ShanghaiTech policy must retain source points and cardinality."""
    points = np.array([[-5.0, 20.0], [450.0, 100.0]], dtype=np.float32)
    allowed = validate_point_annotations(
        points,
        source="official_sha",
        coordinate_base=0,
        image_shape=(400, 300),
        bounds_policy="allow",
    )
    np.testing.assert_array_equal(allowed, points)


def test_continuous_support_boundary_binning():
    """Points at support boundaries [-0.5, W-0.5] must bin into correct stride-4 cells without dropping."""
    H, W = 256, 256
    pts = torch.tensor([
        [-0.49, -0.49],       # Extreme top-left -> cell (0, 0)
        [0.0, 0.0],           # Center of pixel 0 -> cell (0, 0)
        [3.49, 3.49],         # Inside cell 0 -> cell (0, 0)
        [3.51, 3.51],         # Crosses into cell 1 -> cell (1, 1)
        [255.0, 255.0],       # Center of last pixel -> cell (63, 63)
        [255.49, 255.49],     # Extreme bottom-right -> cell (63, 63)
    ], dtype=torch.float32)

    y4 = points_to_y4(pts, H=H, W=W)
    assert y4.sum().item() == 6.0, f"Expected 6 points binned, got {y4.sum().item()}"
    assert y4[0, 0, 0].item() == 3.0   # 3 points in cell (0, 0)
    assert y4[0, 1, 1].item() == 1.0   # 1 point in cell (1, 1)
    assert y4[0, 63, 63].item() == 2.0 # 2 points in cell (63, 63)


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


def test_nwpu_test_never_uses_train_split():
    """NWPU split resolver must correctly map train/val/test without cross-split fallback."""
    cfg = {
        "train_split_file": "train.txt",
        "val_split_file": "val.txt",
        "test_split_file": "test.txt",
    }
    assert resolve_nwpu_split_file(cfg, "train") == "train.txt"
    assert resolve_nwpu_split_file(cfg, "val") == "val.txt"
    assert resolve_nwpu_split_file(cfg, "test") == "test.txt"

    with pytest.raises(ValueError, match="Unsupported NWPU split"):
        resolve_nwpu_split_file(cfg, "unknown_split")
