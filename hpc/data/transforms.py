import io
import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class GeometricTransformResult(dict):
    """Mapping result that also supports the legacy ``image, points = out`` API."""

    def __iter__(self):
        yield self["image"]
        yield self["points"]


# ---------------------------------------------------------------------------
# Scale-Aware Safe Geometric Transforms (SR48 §10)
# ---------------------------------------------------------------------------

class ScaleAwareSafeGeometricTransforms:
    """Isotropic scale + safe-crop + scale-aware crop sampling + horizontal flip.

    Replaces the old ``GeometricTransforms`` with three key additions:

    1. **Safe crop**: reject candidates that cut through a guard radius around
       any annotated point centre (prevents artificial false-negative partial
       persons at crop edges).

    2. **Scale-aware crop mixture**:
       - 75% uniform safe random crop
       - 15% crop centred on a large/isolated point (d_nn ≥ large_nn_threshold)
       - 10% crop centred on a true-image-border point (d_border ≤ border_thresh)

    3. **Special-point metadata**: returns per-point flags
       ``point_large_flags`` and ``point_true_border_flags`` (before crop
       filtering) so the dataset can build fixed ``gt_large_mask16`` /
       ``gt_true_border_mask16`` tensors from surviving crop points.

    Note: d_nn is a **scale proxy** (nearest-neighbour distance), not a true
    head-size measurement.
    """

    def __init__(
        self,
        crop_size: int = 448,
        scale_range: Tuple[float, float] = (0.75, 2.0),
        flip_prob: float = 0.5,
        # Safe crop
        max_crop_attempts: int = 20,
        crop_guard_nn_factor: float = 0.20,
        crop_guard_min_px: float = 8.0,
        crop_guard_max_px: float = 48.0,
        # Scale-aware mixture probabilities
        random_crop_prob: float = 0.75,
        large_center_crop_prob: float = 0.15,
        border_center_crop_prob: float = 0.10,
        # Thresholds for large / border classification
        large_nn_threshold_px: float = 48.0,
        true_border_threshold_px: float = 32.0,
        crop_sampling: str = "safe_mixture",
        compute_point_metadata: bool = True,
    ):
        if crop_size <= 0:
            raise ValueError("crop_size must be > 0")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError(f"Invalid scale_range={scale_range}")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be in [0, 1]")

        self.crop_size = int(crop_size)
        self.scale_range = tuple(float(v) for v in scale_range)
        self.flip_prob = float(flip_prob)
        self.max_crop_attempts = int(max_crop_attempts)
        self.crop_guard_nn_factor = float(crop_guard_nn_factor)
        self.crop_guard_min_px = float(crop_guard_min_px)
        self.crop_guard_max_px = float(crop_guard_max_px)
        self.random_crop_prob = float(random_crop_prob)
        self.large_center_crop_prob = float(large_center_crop_prob)
        self.border_center_crop_prob = float(border_center_crop_prob)
        self.large_nn_threshold_px = float(large_nn_threshold_px)
        self.true_border_threshold_px = float(true_border_threshold_px)
        if crop_sampling not in {"uniform", "safe_mixture"}:
            raise ValueError("crop_sampling must be 'uniform' or 'safe_mixture'")
        self.crop_sampling = crop_sampling
        self.compute_point_metadata = bool(compute_point_metadata)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_nn_distances(pts: np.ndarray, H: int, W: int) -> np.ndarray:
        """Nearest-neighbour distances for each point (N,).

        Single point → max(H, W) per spec §10.2.
        """
        n = len(pts)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        if n == 1:
            return np.array([float(max(H, W))], dtype=np.float32)
        # Pairwise L2; vectorized for speed on typical crowd sizes.
        x = pts[:, 0:1] - pts[:, 0]      # (N, N)
        y = pts[:, 1:2] - pts[:, 1]
        dist = np.sqrt(x ** 2 + y ** 2)  # (N, N)
        np.fill_diagonal(dist, np.inf)
        return dist.min(axis=1).astype(np.float32)

    @staticmethod
    def _compute_border_distances(pts: np.ndarray, H: int, W: int) -> np.ndarray:
        """Distance from each point to the nearest true image border."""
        if len(pts) == 0:
            return np.empty(0, dtype=np.float32)
        x, y = pts[:, 0], pts[:, 1]
        return np.minimum.reduce([x, y, W - 1 - x, H - 1 - y]).astype(np.float32)

    def _guard_radii(self, d_nn: np.ndarray) -> np.ndarray:
        """Per-point guard radius = clip(0.20 * d_nn, 8, 48) px."""
        return np.clip(self.crop_guard_nn_factor * d_nn,
                       self.crop_guard_min_px,
                       self.crop_guard_max_px).astype(np.float32)

    def _is_safe_crop(
        self,
        pts: np.ndarray,
        radii: np.ndarray,
        x0: int,
        y0: int,
    ) -> bool:
        """Return True if no annotated point centre is outside the crop but
        inside its guard-expanded region (artificial border false-negative)."""
        if len(pts) == 0:
            return True
        cs = self.crop_size
        x1, y1 = x0 + cs, y0 + cs
        px, py = pts[:, 0], pts[:, 1]
        inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
        near = (
            (px >= x0 - radii) & (px < x1 + radii) &
            (py >= y0 - radii) & (py < y1 + radii)
        )
        bad = (~inside) & near
        return not bad.any()

    def _candidate_violation_count(
        self,
        pts: np.ndarray,
        radii: np.ndarray,
        x0: int,
        y0: int,
    ) -> int:
        """Count violating points (for soft fallback selection)."""
        if len(pts) == 0:
            return 0
        cs = self.crop_size
        x1, y1 = x0 + cs, y0 + cs
        px, py = pts[:, 0], pts[:, 1]
        inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
        near = (
            (px >= x0 - radii) & (px < x1 + radii) &
            (py >= y0 - radii) & (py < y1 + radii)
        )
        return int((~inside & near).sum())

    def _safe_random_crop(
        self,
        pts: np.ndarray,
        radii: np.ndarray,
        max_x: int,
        max_y: int,
    ) -> Tuple[int, int]:
        """Try max_crop_attempts random origins; return safest if none are safe."""
        best_x, best_y = 0, 0
        best_violations = int(1e9)
        for _ in range(self.max_crop_attempts):
            cx = random.randint(0, max_x) if max_x > 0 else 0
            cy = random.randint(0, max_y) if max_y > 0 else 0
            if self._is_safe_crop(pts, radii, cx, cy):
                return cx, cy
            v = self._candidate_violation_count(pts, radii, cx, cy)
            if v < best_violations:
                best_violations, best_x, best_y = v, cx, cy
        return best_x, best_y

    def _center_crop_on_point(
        self,
        pts: np.ndarray,
        radii: np.ndarray,
        max_x: int,
        max_y: int,
        point_idx: int,
    ) -> Tuple[int, int]:
        """Crop centred on a selected point (clamped to valid range), with safe-crop retry."""
        px, py = float(pts[point_idx, 0]), float(pts[point_idx, 1])
        cx0 = int(round(px - self.crop_size / 2.0))
        cy0 = int(round(py - self.crop_size / 2.0))
        cx0 = max(0, min(cx0, max_x))
        cy0 = max(0, min(cy0, max_y))
        if self._is_safe_crop(pts, radii, cx0, cy0):
            return cx0, cy0
        # Fallback: try a few random crops around the point
        for _ in range(max(1, self.max_crop_attempts // 2)):
            jx = random.randint(0, max_x) if max_x > 0 else 0
            jy = random.randint(0, max_y) if max_y > 0 else 0
            if self._is_safe_crop(pts, radii, jx, jy):
                return jx, jy
        return cx0, cy0  # best-effort

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: Image.Image,
        points: np.ndarray,
    ) -> Dict:
        """Apply transforms and return a metadata dict.

        Returns:
            image: cropped PIL Image (crop_size × crop_size)
            points: (M, 2) float32 array of surviving annotated point centres
            point_large_flags: (M,) bool — proxy for large/isolated person
            point_true_border_flags: (M,) bool — point near true image border
        """
        w, h = image.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size {(w, h)}")

        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

        # --- Isotropic scale ---
        sampled_scale = random.uniform(*self.scale_range)
        min_scale_to_fit = max(self.crop_size / float(w), self.crop_size / float(h))
        scale = max(sampled_scale, min_scale_to_fit)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if len(pts):
            pts = pts.copy()
            pts[:, 0] *= new_w / float(w)
            pts[:, 1] *= new_h / float(h)

        # --- Pre-crop point metadata (in scaled image space) ---
        if self.compute_point_metadata:
            d_nn = self._compute_nn_distances(pts, new_h, new_w)
            d_border = self._compute_border_distances(pts, new_h, new_w)
            radii = self._guard_radii(d_nn) if len(pts) > 0 else np.empty(0, dtype=np.float32)
        else:
            d_nn = np.zeros(len(pts), dtype=np.float32)
            d_border = np.zeros(len(pts), dtype=np.float32)
            radii = np.zeros(len(pts), dtype=np.float32)

        if self.compute_point_metadata and len(d_nn) > 0:
            top25_thresh = np.percentile(d_nn, 75)
            large_flags_scaled = (d_nn >= self.large_nn_threshold_px) | (d_nn >= top25_thresh)
        else:
            large_flags_scaled = np.zeros(len(pts), dtype=bool)

        true_border_flags_scaled = (
            d_border <= self.true_border_threshold_px
            if self.compute_point_metadata and len(d_border) > 0
            else np.zeros(len(pts), dtype=bool)
        )

        # --- Scale-aware crop selection ---
        max_x = new_w - self.crop_size
        max_y = new_h - self.crop_size

        mode_draw = random.random()
        crop_x, crop_y = 0, 0

        large_indices = (
            np.where(large_flags_scaled)[0].tolist() if len(large_flags_scaled) > 0 else []
        )
        border_indices = (
            np.where(true_border_flags_scaled)[0].tolist() if len(true_border_flags_scaled) > 0 else []
        )

        if self.crop_sampling == "uniform":
            crop_x = random.randint(0, max_x) if max_x > 0 else 0
            crop_y = random.randint(0, max_y) if max_y > 0 else 0
        elif mode_draw < self.large_center_crop_prob and large_indices:
            # 15%: centre on a randomly chosen large/isolated point
            idx = random.choice(large_indices)
            crop_x, crop_y = self._center_crop_on_point(pts, radii, max_x, max_y, idx)

        elif mode_draw < self.large_center_crop_prob + self.border_center_crop_prob and border_indices:
            # 10%: centre on a randomly chosen true-border point
            idx = random.choice(border_indices)
            crop_x, crop_y = self._center_crop_on_point(pts, radii, max_x, max_y, idx)

        else:
            # 75%: safe random crop
            crop_x, crop_y = self._safe_random_crop(pts, radii, max_x, max_y)

        image = image.crop((crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size))

        # --- Crop points + flags ---
        if len(pts):
            pts_c = pts.copy()
            pts_c[:, 0] -= crop_x
            pts_c[:, 1] -= crop_y
            valid = (
                (pts_c[:, 0] >= 0) & (pts_c[:, 0] < self.crop_size) &
                (pts_c[:, 1] >= 0) & (pts_c[:, 1] < self.crop_size)
            )
            crop_pts = pts_c[valid]
            crop_large_flags = large_flags_scaled[valid]
            crop_border_flags = true_border_flags_scaled[valid]
        else:
            crop_pts = np.empty((0, 2), dtype=np.float32)
            crop_large_flags = np.empty(0, dtype=bool)
            crop_border_flags = np.empty(0, dtype=bool)

        # --- Horizontal flip ---
        if random.random() < self.flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(crop_pts):
                crop_pts[:, 0] = (self.crop_size - 1.0) - crop_pts[:, 0]
            # Flags remain attached to same points; only coordinates change.

        # Recompute d_nn for surviving crop points (neighbours may have changed after crop)
        if not self.compute_point_metadata:
            crop_dnn = np.zeros(len(crop_pts), dtype=np.float32)
        elif len(crop_pts) > 1:
            crop_dnn = self._compute_nn_distances(crop_pts, self.crop_size, self.crop_size)
        elif len(crop_pts) == 1:
            crop_dnn = np.array([float(max(self.crop_size, self.crop_size))], dtype=np.float32)
        else:
            crop_dnn = np.empty(0, dtype=np.float32)

        return GeometricTransformResult({
            "image": image,
            "points": crop_pts.astype(np.float32, copy=False),
            "point_large_flags": crop_large_flags,
            "point_true_border_flags": crop_border_flags,
            "point_dnn": crop_dnn,
        })


# ---------------------------------------------------------------------------
# Photometric Transforms (unchanged from original)
# ---------------------------------------------------------------------------

class PhotometricTransforms:
    """Geometry-preserving adverse-condition photometric degradations."""

    def __init__(
        self,
        brightness: float = 0.20,
        contrast: float = 0.20,
        saturation: float = 0.15,
        blur_prob: float = 0.20,
        gamma_min: float = 0.35,
        gamma_max: float = 1.80,
        salt_pepper_prob: float = 0.15,
        jpeg_prob: float = 0.20,
        noise_prob: float = 0.20,
        **kwargs,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.blur_prob = float(blur_prob)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.jpeg_prob = float(jpeg_prob)
        self.noise_prob = float(noise_prob)
        if not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("Invalid gamma range")
        for name in ("blur_prob", "salt_pepper_prob", "jpeg_prob", "noise_prob"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        if self.brightness > 0 and random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(
                max(1.0 + random.uniform(-self.brightness, self.brightness), 0.1)
            )
        if self.contrast > 0 and random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(
                max(1.0 + random.uniform(-self.contrast, self.contrast), 0.1)
            )
        if self.saturation > 0 and random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(
                max(1.0 + random.uniform(-self.saturation, self.saturation), 0.0)
            )
        if random.random() < 0.3:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = np.power(np.clip(arr, 1e-6, 1.0), gamma)
            image = Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)))
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        if random.random() < self.noise_prob:
            arr = np.asarray(image, dtype=np.float32)
            arr += np.random.normal(0.0, random.uniform(5.0, 20.0), arr.shape)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if random.random() < self.salt_pepper_prob:
            arr = np.array(image, copy=True)
            hh, ww = arr.shape[:2]
            amount = random.uniform(0.0005, 0.003)
            n_each = int(amount * hh * ww * 0.5)
            if n_each > 0:
                ys = np.random.randint(0, hh, n_each)
                xs = np.random.randint(0, ww, n_each)
                arr[ys, xs] = 255
                ys = np.random.randint(0, hh, n_each)
                xs = np.random.randint(0, ww, n_each)
                arr[ys, xs] = 0
            image = Image.fromarray(arr)
        if random.random() < self.jpeg_prob:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=random.randint(30, 80))
            buffer.seek(0)
        return image


# Backward-compatible alias
GeometricTransforms = ScaleAwareSafeGeometricTransforms
