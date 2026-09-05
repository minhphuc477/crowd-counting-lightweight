from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


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
    valid = (x >= 0) & (x < image_w) & (y >= 0) & (y < image_h)
    if not valid.any():
        return out
    x, y = x[valid], y[valid]
    j = torch.floor((x + 0.5) / stride).long()
    i = torch.floor((y + 0.5) / stride).long()
    valid_cell = (i >= 0) & (i < gh) & (j >= 0) & (j < gw)
    i, j = i[valid_cell], j[valid_cell]
    flat = i * gw + j
    out.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return out


def _pad_to_crop(image: torch.Tensor, points: torch.Tensor, crop_h: int, crop_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = image.shape
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h or pad_w:
        # P1 #2 fix: pad with ImageNet mean values in [0,1] space (BEFORE normalization).
        #
        # Why NOT reflect: reflect padding copies crowd image content without copying the
        # corresponding point annotations → padded area shows visual crowd but GT=0.
        # This creates label noise: the model sees "people" with zero density target.
        #
        # Why NOT zero: raw 0.0 → after normalize_image subtracts ImageNet mean (~0.485/R),
        # padding becomes strongly negative (~-2.1 for R channel), creating large feature
        # activations at borders that the model must learn to ignore.
        #
        # ImageNet mean [0.485, 0.456, 0.406] → after normalization becomes exactly 0.0
        # for all channels → padded cells produce near-zero features and near-zero
        # density prediction, which is correct (no people in the padded region).
        #
        # image is in [0,1] raw tensor space here (normalize_image is called AFTER this).
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        new_h = h + pad_h
        new_w = w + pad_w
        canvas = mean.expand(3, new_h, new_w).clone()
        canvas[:, :h, :w] = image
        image = canvas
    return image, points





def train_transform(
    image: Image.Image,
    points_xy: torch.Tensor,
    crop_size: int = 512,
    scale_range: tuple[float, float] = (0.75, 1.25),
    hflip_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric augmentation that keeps point coordinates exact."""
    image_t = TF.to_tensor(image)
    pts = points_xy.clone().float()

    scale = random.uniform(*scale_range)
    h0, w0 = image_t.shape[-2:]
    h1 = max(32, int(round(h0 * scale)))
    w1 = max(32, int(round(w0 * scale)))
    image_t = TF.resize(image_t, [h1, w1], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if pts.numel():
        pts[:, 0] *= w1 / w0
        pts[:, 1] *= h1 / h0

    image_t, pts = _pad_to_crop(image_t, pts, crop_size, crop_size)
    _, h, w = image_t.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    image_t = image_t[:, top:top + crop_size, left:left + crop_size]
    if pts.numel():
        pts[:, 0] -= left
        pts[:, 1] -= top
        keep = (
            (pts[:, 0] >= 0) & (pts[:, 0] < crop_size) &
            (pts[:, 1] >= 0) & (pts[:, 1] < crop_size)
        )
        pts = pts[keep]

    if random.random() < hflip_prob:
        image_t = torch.flip(image_t, dims=[-1])
        if pts.numel():
            pts[:, 0] = (crop_size - 1) - pts[:, 0]

    # Lightweight photometric augmentation.
    if random.random() < 0.5:
        image_t = TF.adjust_brightness(image_t, random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        image_t = TF.adjust_contrast(image_t, random.uniform(0.85, 1.15))

    return image_t.clamp(0, 1), pts


def normalize_image(image_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    return (image_t - mean) / std


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
    ):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.train = train
        self.output_stride = int(output_stride)
        self.crop_size = int(crop_size)
        self.scale_range = scale_range
        with self.manifest.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        pts = torch.tensor(item.get("points", []), dtype=torch.float32).reshape(-1, 2)

        if self.train:
            image_t, pts = train_transform(
                image, pts,
                crop_size=self.crop_size,
                scale_range=self.scale_range,
            )
        else:
            image_t = TF.to_tensor(image)

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
        "id": [b["id"] for b in batch],
    }


def collate_eval(batch: list[dict]) -> list[dict]:
    # Full-resolution images may differ in shape; evaluate sample-by-sample.
    return batch
