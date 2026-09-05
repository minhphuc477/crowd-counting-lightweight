from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _extract_points(path: Path, dataset: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        pts = np.load(path)
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points", data.get("annPoints", []))
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if path.suffix.lower() == ".txt":
        pts = np.loadtxt(path)
        return np.asarray(pts, dtype=np.float32).reshape(-1, 2)

    mat = loadmat(path)
    if "annPoints" in mat:
        pts = np.asarray(mat["annPoints"], dtype=np.float32).reshape(-1, 2)
    elif "image_info" in mat:  # ShanghaiTech format
        pts = np.asarray(mat["image_info"][0, 0][0, 0][0], dtype=np.float32).reshape(-1, 2)
    elif "points" in mat:
        pts = np.asarray(mat["points"], dtype=np.float32).reshape(-1, 2)
    else:
        # Conservative fallback: only accept an obvious Nx2 numeric array.
        candidates = []
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            a = np.asarray(v)
            if np.issubdtype(a.dtype, np.number) and a.ndim == 2 and a.shape[1] == 2:
                candidates.append((k, a))
        if len(candidates) == 1:
            pts = candidates[0][1].astype(np.float32)
        else:
            raise RuntimeError(f"Could not uniquely identify Nx2 points in {path}; keys={list(mat.keys())}")

    if dataset == "qnrf":
        # QNRF Matlab annotations are 1-indexed; convert to standard 0-indexed pixel coordinates
        pts = pts - 1.0
    return pts


def annotation_for(image: Path, ann_dir: Path, dataset: str) -> Path:
    stem = image.stem
    candidates: list[Path] = []
    if dataset.startswith("sha"):
        candidates += [ann_dir / f"GT_{stem}.mat", ann_dir / f"{stem}.mat"]
    elif dataset == "qnrf":
        candidates += [ann_dir / f"{stem}_ann.mat", ann_dir / f"{stem}.mat"]
    elif dataset == "nwpu":
        candidates += [
            ann_dir / f"{stem}.mat",
            ann_dir / f"{stem}_ann.mat",
            ann_dir / f"{stem}.npy",
            ann_dir / f"{stem}.json",
            ann_dir / f"{stem}.txt",
        ]
    else:
        candidates += [ann_dir / f"{stem}.mat"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No annotation for {image}; tried {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--dataset", choices=["sha_a", "sha_b", "qnrf", "nwpu"], required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--relative-to", type=Path, default=None, help="Root dir to make image paths relative to")
    args = ap.parse_args()

    images = sorted([p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rel_root = args.relative_to.resolve() if args.relative_to else args.out.parent.resolve()

    with args.out.open("w", encoding="utf-8") as f:
        for image in images:
            ann = annotation_for(image, args.annotations, args.dataset)
            pts = _extract_points(ann, args.dataset)
            try:
                img_ref = str(image.resolve().relative_to(rel_root)).replace("\\", "/")
            except ValueError:
                img_ref = str(image.resolve()).replace("\\", "/")
            row = {"image": img_ref, "points": pts.tolist(), "id": image.stem}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(images)} samples -> {args.out}")


if __name__ == "__main__":
    main()
