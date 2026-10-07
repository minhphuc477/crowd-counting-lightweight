from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
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
    image.close()
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
        image.close()
        image = padded
        if pts.numel():
            pts[:, 0] += pad_l
            pts[:, 1] += pad_t
        w1, h1 = image.size

    top = random.randint(0, h1 - crop_size)
    left = random.randint(0, w1 - crop_size)
    image_crop = image.crop((left, top, left + crop_size, top + crop_size))
    image.close()
    image = image_crop

    if pts.numel():
        pts[:, 0] -= left
        pts[:, 1] -= top
        bm = float(boundary_margin)
        keep = (
            (pts[:, 0] >= -bm) & (pts[:, 0] < float(crop_size) + bm) &
            (pts[:, 1] >= -bm) & (pts[:, 1] < float(crop_size) + bm)
        )
        pts = pts[keep]
        # Invariant: coordinates within the crop (including boundary heads) are clamped
        # to the closed support [0.0, crop_size - 1.0].
        pts[:, 0] = pts[:, 0].clamp(0.0, float(crop_size - 1))
        pts[:, 1] = pts[:, 1].clamp(0.0, float(crop_size - 1))

    if hflip_prob > 0.0 and random.random() < hflip_prob:
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image.close()
        image = flipped
        if pts.numel():
            pts[:, 0] = (float(crop_size) - 1.0) - pts[:, 0]
            pts[:, 0] = pts[:, 0].clamp(0.0, float(crop_size - 1))

    image_t = TF.to_tensor(image)
    image.close()

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


def resolve_manifest_path(manifest_val: str | Path, data_root: str | Path | None = None) -> Path | None:
    """Resolve manifest file on local filesystem.

    Tries in order:
    1. Direct path (manifest_val)
    2. Relative to data_root (if provided): data_root / manifest_val
    3. data_root / manifest_val.name
    4. Fallback to repo 'data/' directory (handles cross-environment simulation)
    """
    p = Path(manifest_val)
    if p.is_file():
        return p
    if data_root is not None:
        p_data = Path(data_root) / p
        if p_data.is_file():
            return p_data
        p_name = Path(data_root) / p.name
        if p_name.is_file():
            return p_name
    p_repo_data = Path("data") / p.name
    if p_repo_data.is_file():
        return p_repo_data
    return None


class CrowdManifestDataset(Dataset):
    """Standardized Crowd Counting Dataset with Zero-Stall In-Memory RAM Caching."""

    def __init__(
        self,
        manifest: str | Path, train: bool,
        output_stride: int = 4, crop_size: int = 512,
        scale_range: tuple[float, float] = (0.75, 1.25), hflip_prob: float = 0.5,
        brightness_jitter: float = 0.0, contrast_jitter: float = 0.0,
        gamma_jitter: tuple[float, float] = (1.0, 1.0), random_invert_prob: float = 0.0,
        pad_small_images: bool = False, boundary_margin: float = 0.0, max_size: int = 2048,
        data_root: str | Path | None = None, cache_images: bool = True, preload: bool = False,
    ):
        manifest_path = Path(manifest)
        if manifest_path.name in ("sha_a_train.jsonl", "sha_a_val.jsonl"):
            raise ValueError(f"Forbidden legacy split '{manifest}' under Zero Ad-hoc Split Policy. Use canonical benchmarks.")

        resolved_manifest = resolve_manifest_path(manifest, data_root=data_root)
        self.manifest = resolved_manifest if resolved_manifest is not None else Path(manifest)
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest} (requested '{manifest}')")
        self.root = Path(data_root) if data_root is not None else self.manifest.parent
        self.train = train
        self.output_stride = int(output_stride)
        self.crop_size = int(crop_size)
        self.scale_range = tuple(scale_range)
        self.hflip_prob = float(hflip_prob)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.gamma_jitter = (float(gamma_jitter[0]), float(gamma_jitter[1]))
        self.random_invert_prob = float(random_invert_prob)
        self.pad_small_images = bool(pad_small_images)
        self.boundary_margin = float(boundary_margin)
        self.max_size = int(max_size)
        self.cache_images = bool(cache_images)
        self._raw_bytes_cache: dict[int, bytes] = {}
        with self.manifest.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        seen_ids, seen_images = set(), set()
        for idx, it in enumerate(self.items):
            sid = str(it.get("id", idx))
            if sid in seen_ids:
                raise ValueError(f"Duplicate sample ID '{sid}' in manifest '{self.manifest}' at index {idx}.")
            seen_ids.add(sid)
            img_raw = it.get("image", "")
            if img_raw:
                img_p = Path(img_raw).as_posix()
                if img_p in seen_images:
                    raise ValueError(f"Duplicate image path '{img_p}' in manifest '{self.manifest}' at index {idx}.")
                seen_images.add(img_p)

        # Enforce canonical benchmark partitions
        canonical_counts = {
            "sha_a_train_all.jsonl": 300, "sha_a_test.jsonl": 182,
            "shb_train_all.jsonl": 400, "shb_test.jsonl": 316,
            "ucf_cc_50_all.jsonl": 50, "qnrf_train.jsonl": 1201, "qnrf_test.jsonl": 334,
            "nwpu_train.jsonl": 3109, "nwpu_val.jsonl": 500, "nwpu_test.jsonl": 1500,
        }
        if self.manifest.name in canonical_counts and len(self.items) != canonical_counts[self.manifest.name]:
            raise ValueError(
                f"Partition '{self.manifest.name}' must contain exactly {canonical_counts[self.manifest.name]} images, got {len(self.items)}."
            )

        # Pre-convert coordinate points to contiguous float32 tensors once to avoid
        # per-epoch Python object creation and Linux Copy-On-Write page duplication across workers.
        for it in self.items:
            pts_raw = it.get("points", [])
            it["points_tensor"] = torch.tensor(pts_raw, dtype=torch.float32).reshape(-1, 2)

        if self.cache_images and preload:
            for i, it in enumerate(self.items):
                p = Path(it["image"])
                if not p.is_absolute():
                    p = self.root / p
                self._raw_bytes_cache[i] = p.read_bytes()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        if self.cache_images:
            if idx not in self._raw_bytes_cache:
                self._raw_bytes_cache[idx] = path.read_bytes()
            with io.BytesIO(self._raw_bytes_cache[idx]) as buf, Image.open(buf) as raw_img:
                image = raw_img.convert("RGB")
        else:
            with Image.open(path) as img:
                image = img.convert("RGB")
        pts_tensor = item.get("points_tensor")
        if pts_tensor is not None:
            pts = pts_tensor.clone()
        else:
            pts = torch.tensor(item.get("points", []), dtype=torch.float32).reshape(-1, 2)

        if self.train:
            image_t, pts = train_transform(
                image, pts,
                crop_size=self.crop_size,
                scale_range=self.scale_range,
                hflip_prob=self.hflip_prob,
                brightness_jitter=self.brightness_jitter,
                contrast_jitter=self.contrast_jitter,
                gamma_jitter=self.gamma_jitter,
                random_invert_prob=self.random_invert_prob,
                pad_small_images=self.pad_small_images,
                boundary_margin=self.boundary_margin,
                max_size=self.max_size,
            )
        else:
            image, pts = cap_image_resolution(image, pts, max_size=self.max_size)
            image_t = TF.to_tensor(image)
            image.close()

        h, w = image_t.shape[-2:]
        target_y = rasterize_points(pts, h, w, stride=self.output_stride)
        image_t = normalize_image(image_t)
        return {
            "image": image_t,
            "target_y": target_y,
            "points": pts,
            "id": item.get("id", path.stem),
            "path": str(path),
            "height": h,
            "width": w,
        }


def collate_train(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], 0),
        "target_y": torch.stack([b["target_y"] for b in batch], 0),
        "points": [b["points"] for b in batch],
        "id": [b["id"] for b in batch],
    }


def collate_eval(batch: list[dict]) -> list[dict]:
    # Full-resolution images may differ in shape; evaluate sample-by-sample.
    return batch


def compute_manifest_density(
    manifest: str | Path,
    output_stride: int = 4,
    default_m0: float = 0.015763,
    data_root: str | Path | None = None,
) -> float:
    """Empirically compute the mean cell density m0 across training images in a manifest.

    m0 = total_valid_points / total_stride4_cells.
    Points outside [0, w) x [0, h) are filtered identically to rasterize_points.
    """
    manifest_path_obj = Path(manifest) if manifest is not None else None
    if manifest_path_obj is not None and manifest_path_obj.name in ("sha_a_train.jsonl", "sha_a_val.jsonl"):
        raise ValueError(
            f"Ad-hoc split manifest '{manifest}' has been deleted and is strictly forbidden "
            f"under the Zero Ad-hoc Split Policy! Use 'data/sha_a_train_all.jsonl' (300 samples)."
        )

    if manifest is None:
        return default_m0

    resolved_manifest = resolve_manifest_path(manifest, data_root=data_root)
    manifest_path = resolved_manifest if resolved_manifest is not None else Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest '{manifest_path}' does not exist. Cannot compute manifest density."
        )

    root = Path(data_root) if data_root is not None else manifest_path.parent
    total_pts = 0
    total_cells = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pts_list = item.get("points", [])
            img_path = Path(item["image"])
            if not img_path.is_absolute():
                img_path = root / img_path
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image '{img_path}' referenced at line {line_num} in manifest '{manifest_path}' does not exist."
                )
            with Image.open(img_path) as img:
                w, h = img.size
            cells = math.ceil(h / output_stride) * math.ceil(w / output_stride)
            if pts_list:
                pts_arr = torch.tensor(pts_list, dtype=torch.float32).reshape(-1, 2)
                valid = (
                    (pts_arr[:, 0] >= 0) & (pts_arr[:, 0] < w) &
                    (pts_arr[:, 1] >= 0) & (pts_arr[:, 1] < h)
                )
                n_pts = int(valid.sum().item())
            else:
                n_pts = 0
            total_pts += n_pts
            total_cells += cells

    if total_cells <= 0:
        raise ValueError(f"Manifest '{manifest_path}' contains no valid images or cells.")

    return float(total_pts / total_cells)

