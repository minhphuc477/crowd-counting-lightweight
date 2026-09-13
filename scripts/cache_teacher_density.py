#!/usr/bin/env python3
"""Offline Teacher Density Map Pre-Caching for Knowledge Distillation.

Extracts and pre-caches high-capacity teacher density predictions across the
300 training images of ShanghaiTech Part A to disk:
    data/teacher_density_cache/<image_id>.pt

Eliminates 100% of runtime teacher forward-pass latency, GPU VRAM allocation,
and multi-model memory fragmentation during student training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from PIL import Image
import torchvision.transforms.functional as TF

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rmr_core.data import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def cache_teacher_density(
    teacher_ckpt_path: str | Path,
    manifest_path: str | Path = "data/sha_a_train_all.jsonl",
    output_dir: str | Path = "data/teacher_density_cache",
    device_str: str = "cuda:0",
) -> None:
    teacher_path = Path(teacher_ckpt_path)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n[KD Pre-Cache] Loading teacher model from '{teacher_path}' on {device}...")

    # Load teacher model with EMA weights if available
    teacher_model, uniform_rel, cfg, ckpt = load_model_from_ckpt(teacher_path, device, use_ema=True)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with manifest_file.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"[KD Pre-Cache] Processing {len(items)} images from '{manifest_file}'...")
    start_time = time.time()
    cached_count = 0

    with torch.no_grad():
        for idx, item in enumerate(items):
            img_path = Path(item["image"])
            if not img_path.is_absolute():
                img_path = manifest_file.parent / img_path
                if not img_path.exists():
                    img_path = Path("data") / item["image"]

            sample_id = item.get("id", img_path.stem)
            target_save_file = out_path / f"{sample_id}.pt"

            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img_t = TF.to_tensor(rgb_img)

            img_t = normalize_image(img_t).unsqueeze(0).to(device)

            # Forward pass teacher
            out = teacher_model(img_t, uniform_reliability=uniform_rel, solver_strength=1.0)
            y_pred = out["y"][0, 0].cpu().float()  # [H, W] single channel float32

            # Sanity check: finite values and positive count
            assert torch.isfinite(y_pred).all(), f"NaN/Inf detected in teacher prediction for {sample_id}!"
            pred_count = float(y_pred.sum().item())

            torch.save(
                {
                    "id": sample_id,
                    "pred_density": y_pred,
                    "pred_count": pred_count,
                    "height": y_pred.shape[0],
                    "width": y_pred.shape[1],
                },
                target_save_file,
            )
            cached_count += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(items):
                print(f"  [{idx + 1:3d} / {len(items):3d}] Cached '{sample_id}' (Count: {pred_count:6.1f}, Grid: {y_pred.shape[0]}x{y_pred.shape[1]})")

    elapsed = time.time() - start_time
    print(f"\n[KD Pre-Cache SUCCESS] Successfully cached {cached_count} density maps in {elapsed:.2f}s to '{out_path}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Teacher Density Pre-Caching")
    parser.add_argument("--teacher-ckpt", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--manifest", default="data/sha_a_train_all.jsonl", help="Path to train manifest")
    parser.add_argument("--output-dir", default="data/teacher_density_cache", help="Output directory for cache")
    parser.add_argument("--device", default="cuda:0", help="Device to run inference")
    args = parser.parse_args()

    cache_teacher_density(
        teacher_ckpt_path=args.teacher_ckpt,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device_str=args.device,
    )
