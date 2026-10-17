from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .manifests import compute_manifest_density, resolve_manifest_path
from .transforms import (
    cap_image_resolution,
    normalize_image,
    rasterize_points,
    train_transform,
)


class CrowdManifestDataset(Dataset):
    """Standardized Crowd Counting Dataset with Zero-Stall In-Memory RAM Caching."""

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
        pad_small_images: bool = False,
        boundary_margin: float = 0.0,
        max_size: int = 2048,
        saturation_jitter: float = 0.0,
        grayscale_prob: float = 0.0,
        data_root: str | Path | None = None,
        cache_images: bool = True,
        preload: bool = False,
    ):
        manifest_path = Path(manifest)
        if manifest_path.name in ("sha_a_train.jsonl", "sha_a_val.jsonl"):
            raise ValueError(
                f"Forbidden legacy split '{manifest}' under Zero Ad-hoc Split Policy. "
                "Use canonical benchmarks."
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
        self.pad_small_images = bool(pad_small_images)
        self.boundary_margin = float(boundary_margin)
        self.max_size = int(max_size)
        self.saturation_jitter = float(saturation_jitter)
        self.grayscale_prob = float(grayscale_prob)
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

        canonical_counts = {
            "sha_a_train_all.jsonl": 300,
            "sha_a_test.jsonl": 182,
            "shb_train_all.jsonl": 400,
            "shb_test.jsonl": 316,
            "ucf_cc_50_all.jsonl": 50,
            "qnrf_train.jsonl": 1201,
            "qnrf_test.jsonl": 334,
            "nwpu_train.jsonl": 3109,
            "nwpu_val.jsonl": 500,
            "nwpu_test.jsonl": 1500,
        }
        if self.manifest.name in canonical_counts and len(self.items) != canonical_counts[self.manifest.name]:
            expected_n = canonical_counts[self.manifest.name]
            raise ValueError(
                f"Partition '{self.manifest.name}' must contain exactly {expected_n} images, got {len(self.items)}."
            )

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
                image,
                pts,
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
                saturation_jitter=self.saturation_jitter,
                grayscale_prob=self.grayscale_prob,
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
    return batch


__all__ = [
    "CrowdManifestDataset",
    "cap_image_resolution",
    "collate_eval",
    "collate_train",
    "compute_manifest_density",
    "normalize_image",
    "rasterize_points",
    "resolve_manifest_path",
    "train_transform",
]
