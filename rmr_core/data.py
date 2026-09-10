from __future__ import annotations

import json
import math
import random
import warnings
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
        i = floor((y + 0.5) / stride)
        j = floor((x + 0.5) / stride)
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

    j = torch.floor((x + 0.5) / stride).long().clamp(0, gw - 1)
    i = torch.floor((y + 0.5) / stride).long().clamp(0, gh - 1)

    flat = i * gw + j
    out.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return out


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric + photometric augmentation that keeps point coordinates exact.

    Memory-efficient: resize and crop performed directly in uint8 PIL space.
    Scale protocol: random scale drawn from [scale_range[0], scale_range[1]], with the
    lower bound increased when necessary to permit a real crop_size x crop_size crop
    without synthetic padding:
        scale_lo = max(scale_range[0], crop_size / min(w0, h0))
        scale = uniform(scale_lo, max(scale_range[1], scale_lo))
    Points are transformed via continuous pixel-center scaling:
        x' = (x + 0.5) * (w1 / w0) - 0.5
        y' = (y + 0.5) * (h1 / h0) - 0.5

    Photometric augmentation (all opt-in, inactive by default):
        brightness_jitter: ±factor uniform, applied with prob 0.5
        contrast_jitter:   ±factor uniform, applied with prob 0.5
        gamma_jitter:      (lo, hi) log-uniform gamma exponent, applied with prob 0.5
        random_invert_prob: probability of pixel-inversion (1.0 - x), models dark/negative images
    """
    pts = points_xy.clone().float()
    w0, h0 = image.size

    min_dim = min(w0, h0)
    min_scale = max(float(scale_range[0]), float(crop_size) / float(min_dim))
    max_scale = max(float(scale_range[1]), min_scale)
    scale = random.uniform(min_scale, max_scale)
    w1 = max(crop_size, int(round(w0 * scale)))
    h1 = max(crop_size, int(round(h0 * scale)))

    if (w1, h1) != (w0, h0):
        image = image.resize((w1, h1), Image.Resampling.BILINEAR)
        if pts.numel():
            pts[:, 0] = (pts[:, 0] + 0.5) * (w1 / w0) - 0.5
            pts[:, 1] = (pts[:, 1] + 0.5) * (h1 / h0) - 0.5

    top = random.randint(0, h1 - crop_size)
    left = random.randint(0, w1 - crop_size)
    image_crop = image.crop((left, top, left + crop_size, top + crop_size))
    image.close()
    image = image_crop

    if pts.numel():
        pts[:, 0] -= left
        pts[:, 1] -= top
        keep = (
            (pts[:, 0] >= 0) & (pts[:, 0] < crop_size) &
            (pts[:, 1] >= 0) & (pts[:, 1] < crop_size)
        )
        pts = pts[keep]

    if hflip_prob > 0.0 and random.random() < hflip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if pts.numel():
            pts[:, 0] = (crop_size - 1) - pts[:, 0]

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
    """Dataset over a standardized JSONL manifest.

    Each line:
      {"image": "relative/or/absolute/path.jpg", "points": [[x,y], ...], "id": "optional"}
    """

    def __init__(
        self,
        manifest: str | Path,
        train: bool,
        output_stride: int = 4,
        crop_size: int = 512,
        scale_range: tuple[float, float] = (0.75, 1.25),
        hflip_prob: float = 0.5,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
        gamma_jitter: tuple[float, float] = (1.0, 1.0),
        random_invert_prob: float = 0.0,
        data_root: str | Path | None = None,
    ):
        manifest_str = str(manifest).replace("\\", "/")
        if manifest_str.endswith("sha_a_train.jsonl") or manifest_str.endswith("sha_a_val.jsonl"):
            raise ValueError(
                f"Ad-hoc split manifest '{manifest}' has been deleted and is strictly forbidden "
                f"under the Zero Ad-hoc Split Policy! Use 'data/sha_a_train_all.jsonl' (300 samples) "
                f"and 'data/sha_a_test.jsonl' (182 samples)."
            )

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
        with self.manifest.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        seen_ids = set()
        seen_images = set()
        for idx, it in enumerate(self.items):
            sid = str(it.get("id", idx))
            if sid in seen_ids:
                raise ValueError(
                    f"Duplicate sample ID '{sid}' in manifest '{self.manifest}' at index {idx}."
                )
            seen_ids.add(sid)
            img_p = str(it.get("image", "")).replace("\\", "/")
            if img_p:
                if img_p in seen_images:
                    raise ValueError(
                        f"Duplicate image path '{img_p}' in manifest '{self.manifest}' at index {idx}."
                    )
                seen_images.add(img_p)

        # Enforce canonical benchmark partitions for ShanghaiTech Part A
        if self.manifest.name == "sha_a_train_all.jsonl" and len(self.items) != 300:
            raise ValueError(
                f"ShanghaiTech Part A train partition must contain exactly 300 images, got {len(self.items)}."
            )
        if self.manifest.name == "sha_a_test.jsonl" and len(self.items) != 182:
            raise ValueError(
                f"ShanghaiTech Part A test partition must contain exactly 182 images, got {len(self.items)}."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as img:
            image = img.convert("RGB")
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
            )
        else:
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
    manifest_str = str(manifest).replace("\\", "/")
    if manifest_str.endswith("sha_a_train.jsonl") or manifest_str.endswith("sha_a_val.jsonl"):
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

