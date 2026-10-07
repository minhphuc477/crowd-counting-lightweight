"""Generate canonical .jsonl manifests for UCF_CC_50 benchmark.

Creates:
- data/ucf_cc_50_all.jsonl (50 images)
- data/ucf_cc_50_fold1_train.jsonl (40 images), data/ucf_cc_50_fold1_test.jsonl (10 images)
- data/ucf_cc_50_fold2_train.jsonl (40 images), data/ucf_cc_50_fold2_test.jsonl (10 images)
- data/ucf_cc_50_fold3_train.jsonl (40 images), data/ucf_cc_50_fold3_test.jsonl (10 images)
- data/ucf_cc_50_fold4_train.jsonl (40 images), data/ucf_cc_50_fold4_test.jsonl (10 images)
- data/ucf_cc_50_fold5_train.jsonl (40 images), data/ucf_cc_50_fold5_test.jsonl (10 images)
"""
from __future__ import annotations

import json
from pathlib import Path
import scipy.io
from PIL import Image


def generate_manifests() -> None:
    data_dir = Path("data")
    ucf_dir = data_dir / "ucf_cc_50" / "UCF_CC_50"
    if not ucf_dir.exists():
        raise FileNotFoundError(f"UCF_CC_50 directory not found at {ucf_dir}")

    all_items = []
    for i in range(1, 51):
        img_p = ucf_dir / f"{i}.jpg"
        mat_p = ucf_dir / f"{i}_ann.mat"
        if not img_p.exists() or not mat_p.exists():
            raise FileNotFoundError(f"Missing image or mat for index {i}: {img_p}, {mat_p}")

        mat = scipy.io.loadmat(str(mat_p))
        ann_points = mat["annPoints"].tolist()
        rel_img_path = f"ucf_cc_50/UCF_CC_50/{i}.jpg"

        item = {
            "image": rel_img_path,
            "points": ann_points,
            "id": f"ucf50_{i:02d}",
        }
        all_items.append(item)

    # 1. Write ucf_cc_50_all.jsonl
    all_out = data_dir / "ucf_cc_50_all.jsonl"
    with all_out.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(all_items)} samples to {all_out}")

    # 2. Write 5-fold cross-validation manifests (10 test images per fold)
    for fold in range(1, 6):
        test_start = (fold - 1) * 10
        test_end = fold * 10
        test_items = all_items[test_start:test_end]
        train_items = all_items[:test_start] + all_items[test_end:]

        tr_out = data_dir / f"ucf_cc_50_fold{fold}_train.jsonl"
        te_out = data_dir / f"ucf_cc_50_fold{fold}_test.jsonl"

        with tr_out.open("w", encoding="utf-8") as f:
            for it in train_items:
                f.write(json.dumps(it) + "\n")
        with te_out.open("w", encoding="utf-8") as f:
            for it in test_items:
                f.write(json.dumps(it) + "\n")

        print(f"Fold {fold}: {len(train_items)} train -> {tr_out.name}, {len(test_items)} test -> {te_out.name}")


if __name__ == "__main__":
    generate_manifests()
