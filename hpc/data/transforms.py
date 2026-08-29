from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image


def compute_isotropic_resize(
    old_w: int,
    old_h: int,
    crop_size: int,
    sampled_scale: float,
) -> Tuple[int, int]:
    """Compute target image size under isotropic scaling ensuring dimensions >= crop_size."""
    fit_scale = max(
        float(crop_size) / float(old_w),
        float(crop_size) / float(old_h),
    )
    scale = max(sampled_scale, fit_scale)
    new_w = max(crop_size, int(round(float(old_w) * scale)))
    new_h = max(crop_size, int(round(float(old_h) * scale)))
    return new_w, new_h


def in_closed_pixel_support(
    points: np.ndarray,
    width: int | float,
    height: int | float,
) -> np.ndarray:
    """Boolean mask of points lying within closed continuous support [-0.5, W-0.5] x [-0.5, H-0.5]."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    return (
        (pts[:, 0] >= -0.5)
        & (pts[:, 0] <= float(width) - 0.5)
        & (pts[:, 1] >= -0.5)
        & (pts[:, 1] <= float(height) - 0.5)
    )


class NTPCGeometricTransform:
    """True isotropic random scale -> uniform random crop -> horizontal flip.

    Point coordinates follow zero-based pixel-center convention where the continuous
    support for an image of width W and height H is:
        [-0.5, W - 0.5] x [-0.5, H - 0.5].
    """

    def __init__(
        self,
        crop_size: int = 256,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        flip_prob: float = 0.5,
    ):
        if crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"Invalid scale_range: {scale_range}")
        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1]")

        self.crop_size = int(crop_size)
        self.scale_range = tuple(float(x) for x in scale_range)
        self.flip_prob = float(flip_prob)

    def __call__(
        self, image: Image.Image, points: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        old_w, old_h = image.size
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()

        # 1. True isotropic scale: enforce single scale factor to prevent aspect ratio distortion
        sampled_scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w, new_h = compute_isotropic_resize(old_w, old_h, self.crop_size, sampled_scale)

        image_scaled = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        sx = new_w / float(old_w)
        sy = new_h / float(old_h)
        if len(pts) > 0:
            pts[:, 0] = (pts[:, 0] + 0.5) * sx - 0.5
            pts[:, 1] = (pts[:, 1] + 0.5) * sy - 0.5

        # 2. Uniform random crop of size (crop_size, crop_size)
        max_x = new_w - self.crop_size
        max_y = new_h - self.crop_size
        crop_x = random.randint(0, max_x) if max_x > 0 else 0
        crop_y = random.randint(0, max_y) if max_y > 0 else 0

        image_crop = image_scaled.crop(
            (crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size)
        )

        if len(pts) > 0:
            pts[:, 0] -= float(crop_x)
            pts[:, 1] -= float(crop_y)

            # Filter points in closed continuous support [-0.5, crop_size - 0.5]
            valid = in_closed_pixel_support(pts, self.crop_size, self.crop_size)
            pts = pts[valid]

        # 3. Random horizontal flip (invariant under reflection [-0.5, W-0.5] <-> [-0.5, W-0.5])
        if random.random() < self.flip_prob:
            image_crop = image_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(pts) > 0:
                pts[:, 0] = (float(self.crop_size) - 1.0) - pts[:, 0]

        # 4. Invariant assertion on transformed points
        if len(pts) > 0:
            if not in_closed_pixel_support(pts, self.crop_size, self.crop_size).all():
                raise RuntimeError(
                    f"Geometric transform produced out-of-bounds points for crop_size={self.crop_size}"
                )

        return image_crop, pts
