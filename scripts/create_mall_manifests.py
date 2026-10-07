"""Generate canonical .jsonl manifests for Mall dataset.

Creates:
- data/mall_train.jsonl (frames 1-800, standard benchmark training split)
- data/mall_test.jsonl (frames 801-2000, standard benchmark test split)
- data/mall_all.jsonl (all 2,000 frames)
"""
from __future__ import annotations

import json
from pathlib import Path
import scipy.io


def generate_manifests() -> None:
    data_dir = Path("data")
    mall_dir = data_dir / "mall" / "mall_dataset"
    mat_path = mall_dir / "mall_gt.mat"
    frames_dir = mall_dir / "frames"

    if not mat_path.exists() or not frames_dir.exists():
        raise FileNotFoundError(f"Mall dataset files missing in {mall_dir}")

    mat = scipy.io.loadmat(str(mat_path))
    frames = mat["frame"][0]
    total_frames = len(frames)
    if total_frames != 2000:
        raise ValueError(f"Expected 2000 frames, found {total_frames}")

    all_items = []
    for i in range(total_frames):
        frame_idx = i + 1
        img_name = f"seq_{frame_idx:06d}.jpg"
        img_file = frames_dir / img_name
        if not img_file.exists():
            raise FileNotFoundError(f"Frame missing: {img_file}")

        # Head coordinates: array of shape (N, 2)
        loc = frames[i][0][0][0]
        points = loc.tolist() if loc.size > 0 else []

        rel_path = f"mall/mall_dataset/frames/{img_name}"
        item = {
            "image": rel_path,
            "points": points,
            "id": f"mall_{frame_idx:06d}",
        }
        all_items.append(item)

    # 1. Write all
    all_out = data_dir / "mall_all.jsonl"
    with all_out.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(all_items)} samples to {all_out}")

    # 2. Write train (1-800) and test (801-2000)
    train_items = all_items[:800]
    test_items = all_items[800:]

    tr_out = data_dir / "mall_train.jsonl"
    te_out = data_dir / "mall_test.jsonl"

    with tr_out.open("w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")
    with te_out.open("w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it) + "\n")

    print(f"Wrote {len(train_items)} train samples to {tr_out}")
    print(f"Wrote {len(test_items)} test samples to {te_out}")


if __name__ == "__main__":
    generate_manifests()
