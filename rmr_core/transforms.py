from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def rasterize_points(
    points_xy: torch.Tensor,
    image_h: int,
    image_w: int,
    stride: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact stride-cell counts from point annotations.

    Canonical assignment:
        i = floor(y / stride)
        j = floor(x / stride)
    Points outside the actual image support are ignored, never clipped into a border cell.
    """
    gh = math.ceil(image_h / stride)
    gw = math.ceil(image_w / stride)
    out = torch.zeros((1, gh, gw), dtype=dtype)
    if points_xy.numel() == 0:
        return out

    pts = points_xy.float()
    x, y = pts[:, 0], pts[:, 1]
    # Filter truly out-of-image points first (x < 0 or x >= image_w).
    valid = (x >= 0) & (x < image_w) & (y >= 0) & (y < image_h)
    if not valid.any():
        return out
    x, y = x[valid], y[valid]

    j = torch.floor(x / float(stride)).long().clamp(0, gw - 1)
    i = torch.floor(y / float(stride)).long().clamp(0, gh - 1)

    flat = i * gw + j
    out.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return out


def cap_image_resolution(
    image: Image.Image,
    pts: torch.Tensor,
    max_size: int = 2048,
) -> tuple[Image.Image, torch.Tensor]:
    """Downscale image and points if maximum dimension exceeds max_size."""
    w0, h0 = image.size
    max_dim = max(w0, h0)
    if max_dim <= max_size:
        return image, pts
    cap_scale = float(max_size) / float(max_dim)
    w_cap, h_cap = int(round(w0 * cap_scale)), int(round(h0 * cap_scale))
    image_capped = image.resize((w_cap, h_cap), Image.Resampling.BILINEAR)
    if pts.numel():
        pts = pts.clone()
        pts[:, 0] = (pts[:, 0] + 0.5) * (w_cap / w0) - 0.5
        pts[:, 1] = (pts[:, 1] + 0.5) * (h_cap / h0) - 0.5
    return image_capped, pts


def train_transform(
    image: Image.Image,
    points_xy: torch.Tensor,
    crop_size: int = 512,
    scale_range: tuple[float, float] = (0.75, 1.25),
    hflip_prob: float = 0.5,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    gamma_jitter: tuple[float, float] = (1.0, 1.0),
    random_invert_prob: float = 0.0,
    pad_small_images: bool = False,
    boundary_margin: float = 0.0,
    max_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Universal SOTA data augmentation for arbitrary crowd counting datasets (SHA, SHB, QNRF, NWPU, JHU).

    Universal 0.0% padding invariant (pad_small_images=False, default):
        Scale shorter side so min(w1, h1) >= crop_size, guaranteeing that every single pixel
        in the crop is a real photo pixel (0.0% synthetic gray padding).
    Multi-dataset scale cap (max_size=2048):
        Downscales ultra-large images (e.g. 6000x4000 in QNRF/NWPU) to max_size before cropping
        to eliminate RAM explosion and PIL resize latency.
    Continuous pixel-center scaling:
        x' = (x + 0.5) * (w1 / w0) - 0.5, y' = (y + 0.5) * (h1 / h0) - 0.5.
    """
    pts = points_xy.clone().float().reshape(-1, 2)
    image, pts = cap_image_resolution(image, pts, max_size=max_size)
    w0, h0 = image.size
    max_dim, min_dim = max(w0, h0), min(w0, h0)
    if pad_small_images:
        scale_lo = max(1.0, float(scale_range[0])) if min_dim < crop_size else float(scale_range[0])
        scale_hi = max(scale_lo, float(scale_range[1]))
        scale = random.uniform(scale_lo, scale_hi)
        w1 = int(round(w0 * scale))
        h1 = int(round(h0 * scale))
    else:
        # 0.0% synthetic padding: scale shorter side so min(w1, h1) >= crop_size, preserving scale variance
        min_scale = max(float(scale_range[0]), float(crop_size) / float(max(min_dim, 1)))
        max_allowed = max(min_scale * 1.25, float(max_size) / float(max(max_dim, 1)))
        bandwidth_ratio = float(scale_range[1]) / max(float(scale_range[0]), 1e-4)
        target_hi = max(float(scale_range[1]), min_scale * min(bandwidth_ratio, 1.35))
        scale_hi = max(min_scale, min(target_hi, max_allowed))
        scale = random.uniform(min_scale, scale_hi)
        w1 = max(crop_size, int(round(w0 * scale)))
        h1 = max(crop_size, int(round(h0 * scale)))

    if (w1, h1) != (w0, h0):
        image = image.resize((w1, h1), Image.Resampling.BILINEAR)
        if pts.numel():
            pts[:, 0] = (pts[:, 0] + 0.5) * (w1 / w0) - 0.5
            pts[:, 1] = (pts[:, 1] + 0.5) * (h1 / h0) - 0.5

    if pad_small_images and (w1 < crop_size or h1 < crop_size):
        pad_w = max(0, crop_size - w1)
        pad_h = max(0, crop_size - h1)
        padded = Image.new("RGB", (w1 + pad_w, h1 + pad_h), color=(128, 128, 128))
        pad_l, pad_t = pad_w // 2, pad_h // 2
        padded.paste(image, (pad_l, pad_t))
        image = padded
        if pts.numel():
            pts[:, 0] += pad_l
            pts[:, 1] += pad_t
        w1, h1 = image.size

    top = random.randint(0, h1 - crop_size)
    left = random.randint(0, w1 - crop_size)
    image_crop = image.crop((left, top, left + crop_size, top + crop_size))
    image = image_crop

    if pts.numel():
        pts[:, 0] -= left
        pts[:, 1] -= top
        keep = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] < float(crop_size)) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] < float(crop_size))
        )
        pts = pts[keep]
        if pts.numel():
            pts[:, 0] = pts[:, 0].clamp(0.0, float(crop_size - 1))
            pts[:, 1] = pts[:, 1].clamp(0.0, float(crop_size - 1))

    if hflip_prob > 0.0 and random.random() < hflip_prob:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = flipped
        if pts.numel():
            pts[:, 0] = (float(crop_size) - 1.0) - pts[:, 0]
            pts[:, 0] = pts[:, 0].clamp(0.0, float(crop_size - 1))

    image_t = TF.to_tensor(image)

    # Photometric augmentation (opt-in via config; default disabled)
    if brightness_jitter > 0.0 and random.random() < 0.5:
        image_t = TF.adjust_brightness(
            image_t, random.uniform(max(0.0, 1.0 - brightness_jitter), 1.0 + brightness_jitter)
        )
    if contrast_jitter > 0.0 and random.random() < 0.5:
        image_t = TF.adjust_contrast(
            image_t, random.uniform(max(0.0, 1.0 - contrast_jitter), 1.0 + contrast_jitter)
        )
    # Gamma correction: simulates exposure variation and low-light conditions.
    # gamma_jitter = (lo, hi); gamma drawn log-uniformly from [lo, hi].
    # gamma < 1 brightens; gamma > 1 darkens (low-light simulation).
    gamma_lo, gamma_hi = float(gamma_jitter[0]), float(gamma_jitter[1])
    if gamma_lo < gamma_hi and random.random() < 0.5:
        log_gamma = random.uniform(math.log(gamma_lo), math.log(gamma_hi))
        gamma_val = math.exp(log_gamma)
        image_t = TF.adjust_gamma(image_t, gamma=gamma_val)
    # Random invert: simulates negative film / strong contrast reversal.
    if random_invert_prob > 0.0 and random.random() < random_invert_prob:
        image_t = 1.0 - image_t

    return image_t.clamp(0, 1), pts


def normalize_image(image_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    return (image_t - mean) / std


__all__ = [
    "rasterize_points",
    "cap_image_resolution",
    "train_transform",
    "normalize_image",
]
