from __future__ import annotations

import math
import random

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
        pts[:, 0] = ((pts[:, 0] + 0.5) * (w_cap / w0) - 0.5).clamp(0.0, float(w_cap) - 1e-4)
        pts[:, 1] = ((pts[:, 1] + 0.5) * (h_cap / h0) - 0.5).clamp(0.0, float(h_cap) - 1e-4)
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
    saturation_jitter: float = 0.0,
    grayscale_prob: float = 0.0,
    random_erasing_prob: float = 0.0,
    erasing_scale: tuple[float, float] = (0.005, 0.03),
    erasing_ratio: tuple[float, float] = (0.5, 2.0),
    erasing_max_patches: int = 2,
    crowd_aware_crop: bool = True,
    crowd_aware_prob: float = 0.50,
    stratified_crowd_crop: bool = False,
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
    w0, h0 = image.size
    if pts.numel() > 0:
        in_bounds = (
            (pts[:, 0] >= 0.0) & (pts[:, 0] < float(w0)) &
            (pts[:, 1] >= 0.0) & (pts[:, 1] < float(h0))
        )
        pts = pts[in_bounds]

    # Virtual resolution capping: compute effective scale on capped dimension space
    # without allocating redundant multi-megapixel intermediate PIL buffers.
    max_dim = max(w0, h0)
    if max_dim > max_size:
        cap_scale = float(max_size) / float(max_dim)
        w_cap, h_cap = int(round(w0 * cap_scale)), int(round(h0 * cap_scale))
    else:
        w_cap, h_cap = w0, h0
    max_dim_cap, min_dim_cap = max(w_cap, h_cap), min(w_cap, h_cap)

    if pad_small_images:
        scale_lo = float(scale_range[0])
        scale_hi = max(scale_lo, float(scale_range[1]))
        scale = random.uniform(scale_lo, scale_hi)
        w1 = int(round(w_cap * scale))
        h1 = int(round(h_cap * scale))
    else:
        # 0.0% synthetic padding: scale shorter side so min(w1, h1) >= crop_size, preserving scale variance
        min_scale = max(float(scale_range[0]), float(crop_size) / float(max(min_dim_cap, 1)))
        max_allowed = max(min_scale * 1.25, float(max_size) / float(max(max_dim_cap, 1)))
        bandwidth_ratio = float(scale_range[1]) / max(float(scale_range[0]), 1e-4)
        target_hi = max(float(scale_range[1]), min_scale * min(bandwidth_ratio, 1.35))
        scale_hi = max(min_scale, min(target_hi, max_allowed))
        scale = random.uniform(min_scale, scale_hi)
        w1 = max(crop_size, int(round(w_cap * scale)))
        h1 = max(crop_size, int(round(h_cap * scale)))

    scale_x = float(w1) / float(w0)
    scale_y = float(h1) / float(h0)
    if pts.numel():
        pts[:, 0] = ((pts[:, 0] + 0.5) * scale_x - 0.5).clamp(0.0, float(w1) - 1e-4)
        pts[:, 1] = ((pts[:, 1] + 0.5) * scale_y - 0.5).clamp(0.0, float(h1) - 1e-4)

    is_padded = False
    if pad_small_images and (w1 < crop_size or h1 < crop_size):
        work_img = image.resize((w1, h1), Image.Resampling.BILINEAR) if (w1, h1) != (w0, h0) else image
        pad_w = max(0, crop_size - w1)
        pad_h = max(0, crop_size - h1)
        padded = Image.new("RGB", (w1 + pad_w, h1 + pad_h), color=(128, 128, 128))
        pad_l, pad_t = pad_w // 2, pad_h // 2
        padded.paste(work_img, (pad_l, pad_t))
        if work_img is not image:
            work_img.close()
        padded_img = padded
        if pts.numel():
            pts[:, 0] += pad_l
            pts[:, 1] += pad_t
        w1, h1 = padded_img.size
        is_padded = True

    max_left = max(0, w1 - crop_size)
    max_top = max(0, h1 - crop_size)
    if (
        crowd_aware_crop
        and pts.numel() > 0
        and random.random() < float(crowd_aware_prob)
        and (max_left > 0 or max_top > 0)
    ):
        if stratified_crowd_crop:
            # Opt-in 64x64 cell-stratified sampling for extreme multi-scale datasets
            cell_gx = torch.div(pts[:, 0].long(), 64, rounding_mode="floor")
            cell_gy = torch.div(pts[:, 1].long(), 64, rounding_mode="floor")
            cell_ids = cell_gy * 100000 + cell_gx
            unique_cells = torch.unique(cell_ids)
            chosen_cell = unique_cells[random.randint(0, unique_cells.numel() - 1)]
            cell_pts = pts[cell_ids == chosen_cell]
            anchor_idx = random.randint(0, cell_pts.shape[0] - 1)
            anchor_x = float(cell_pts[anchor_idx, 0].item())
            anchor_y = float(cell_pts[anchor_idx, 1].item())
        else:
            # Canonical V64 Champion point-uniform crowd-aware anchor sampling
            anchor_idx = random.randint(0, pts.shape[0] - 1)
            anchor_x = float(pts[anchor_idx, 0].item())
            anchor_y = float(pts[anchor_idx, 1].item())
        jitter_x = random.randint(-crop_size // 4, crop_size // 4)
        jitter_y = random.randint(-crop_size // 4, crop_size // 4)
        left = int(max(0, min(max_left, round(anchor_x - crop_size / 2.0 + jitter_x))))
        top = int(max(0, min(max_top, round(anchor_y - crop_size / 2.0 + jitter_y))))
    else:
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0

    if is_padded:
        cropped = padded_img.crop((left, top, left + crop_size, top + crop_size))
        padded_img.close()
    else:
        src_l = max(0.0, float(left) / scale_x)
        src_t = max(0.0, float(top) / scale_y)
        src_r = min(float(w0), float(left + crop_size) / scale_x)
        src_b = min(float(h0), float(top + crop_size) / scale_y)
        cropped = image.resize((crop_size, crop_size), Image.Resampling.BILINEAR, box=(src_l, src_t, src_r, src_b))

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
        flipped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        cropped.close()
        cropped = flipped
        if pts.numel():
            pts[:, 0] = (float(crop_size) - 1.0) - pts[:, 0]
            pts[:, 0] = pts[:, 0].clamp(0.0, float(crop_size - 1))

    image_t = TF.to_tensor(cropped)
    cropped.close()

    # Photometric augmentation (opt-in via config; default disabled)
    if brightness_jitter > 0.0 and random.random() < 0.5:
        image_t = TF.adjust_brightness(
            image_t, random.uniform(max(0.0, 1.0 - brightness_jitter), 1.0 + brightness_jitter)
        )
    if contrast_jitter > 0.0 and random.random() < 0.5:
        image_t = TF.adjust_contrast(
            image_t, random.uniform(max(0.0, 1.0 - contrast_jitter), 1.0 + contrast_jitter)
        )
    if saturation_jitter > 0.0 and random.random() < 0.5:
        image_t = TF.adjust_saturation(
            image_t, random.uniform(max(0.0, 1.0 - saturation_jitter), 1.0 + saturation_jitter)
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
    # Random grayscale: simulates monochrome photography / illumination invariance.
    if grayscale_prob > 0.0 and random.random() < grayscale_prob:
        image_t = TF.rgb_to_grayscale(image_t, num_output_channels=3)

    # Point-Preserving Random Erasing / Patch Occlusion Augmentation
    image_t, pts = point_preserving_random_erasing(
        image_t,
        pts,
        prob=random_erasing_prob,
        scale=erasing_scale,
        ratio=erasing_ratio,
        max_patches=erasing_max_patches,
    )

    return image_t.clamp(0, 1), pts


def point_preserving_random_erasing(
    image_t: torch.Tensor,
    pts: torch.Tensor,
    prob: float = 0.25,
    scale: tuple[float, float] = (0.005, 0.03),
    ratio: tuple[float, float] = (0.5, 2.0),
    max_patches: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Point-Preserving Random Erasing / Patch Occlusion Augmentation.

    Simulates physical crowd occlusions (umbrellas, shoulders, signs, foreground obstacles).
    Ground-truth points inside erased patches are PRESERVED 100%, forcing the network
    to infer occluded heads from surrounding crowd context and regional density.
    """
    if prob <= 0.0 or random.random() >= prob:
        return image_t, pts
    c, h, w = image_t.shape
    image_t = image_t.clone()
    c_mean = image_t.mean(dim=(-2, -1), keepdim=True)
    s_lo, s_hi = min(float(scale[0]), float(scale[1])), max(float(scale[0]), float(scale[1]))
    r_lo = max(1e-4, min(float(ratio[0]), float(ratio[1])))
    r_hi = max(1e-4, max(float(ratio[0]), float(ratio[1])))
    num_patches = random.randint(1, max(1, max_patches))
    for _ in range(num_patches):
        target_area = float(h * w) * random.uniform(s_lo, s_hi)
        aspect_ratio = math.exp(random.uniform(math.log(r_lo), math.log(r_hi)))
        h_e = int(round(math.sqrt(target_area * aspect_ratio)))
        w_e = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < h_e < h and 0 < w_e < w:
            top_e = random.randint(0, h - h_e)
            left_e = random.randint(0, w - w_e)
            image_t[:, top_e : top_e + h_e, left_e : left_e + w_e] = c_mean
    return image_t, pts


def resolve_backbone_norm_mode(backbone_name: str | None = None, norm_mode: str = "auto") -> str:
    """Resolve canonical input normalization mode ('inception' vs 'imagenet') from backbone family."""
    mode = str(norm_mode or "auto").strip().lower()
    if mode in ("inception", "default", "0.5"):
        return "inception"
    if mode in ("imagenet", "in1k"):
        return "imagenet"
    bname = str(backbone_name or "").strip().lower()
    if not bname or "mobilenetv4" in bname or "inception" in bname:
        return "inception"
    return "imagenet"


def normalize_image(image_t: torch.Tensor, mode: str = "inception") -> torch.Tensor:
    if str(mode).lower() == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    return (image_t - mean) / std


__all__ = [
    "rasterize_points",
    "cap_image_resolution",
    "train_transform",
    "resolve_backbone_norm_mode",
    "normalize_image",
    "point_preserving_random_erasing",
]
