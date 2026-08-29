from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image


class NTPCGeometricTransform:
    """Random scale -> uniform random crop -> horizontal flip.

    Point coordinates are transformed in exact pixel-center zero-based convention:
    bounds [0.0, crop_w - 1.0] x [0.0, crop_h - 1.0].
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

        # 1. Random isotropic scale
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_w = max(self.crop_size, int(round(old_w * scale)))
        new_h = max(self.crop_size, int(round(old_h * scale)))

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

            # Filter points that fell outside the crop window
            valid = (
                (pts[:, 0] >= 0.0)
                & (pts[:, 0] <= float(self.crop_size - 1.0))
                & (pts[:, 1] >= 0.0)
                & (pts[:, 1] <= float(self.crop_size - 1.0))
            )
            pts = pts[valid]

        # 3. Random horizontal flip
        if random.random() < self.flip_prob:
            image_crop = image_crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(pts) > 0:
                pts[:, 0] = float(self.crop_size - 1.0) - pts[:, 0]

        # 4. Invariant assertion on transformed points
        if len(pts) > 0:
            if not (
                np.all(pts[:, 0] >= 0.0)
                and np.all(pts[:, 0] <= float(self.crop_size - 1.0))
                and np.all(pts[:, 1] >= 0.0)
                and np.all(pts[:, 1] <= float(self.crop_size - 1.0))
            ):
                raise RuntimeError(
                    f"Geometric transform produced out-of-bounds points for crop_size={self.crop_size}"
                )

        return image_crop, pts
