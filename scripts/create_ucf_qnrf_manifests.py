"""Extract and generate canonical .jsonl manifests for UCF-QNRF benchmark.

UCF-QNRF benchmark:
- 1,535 total images
- 1,201 Train images in Train/
- 334 Test images in Test/
- Points in img_xxxx_ann.mat -> annPoints

Creates:
- data/qnrf_train.jsonl (1,201 images)
- data/qnrf_test.jsonl (334 images)
- data/qnrf_all.jsonl (1,535 images)
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile
import scipy.io


def extract_if_needed(zip_path: Path, dest_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")

    # Check if already extracted
    extracted_train = dest_dir / "Train"
    if not extracted_train.exists():
        extracted_nested = dest_dir / "UCF-QNRF_ECCV18" / "Train"
        if extracted_nested.exists():
            return dest_dir / "UCF-QNRF_ECCV18"

    if extracted_train.exists():
        print(f"UCF-QNRF already extracted in {dest_dir}")
        return dest_dir

    print(f"Extracting {zip_path} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    nested = dest_dir / "UCF-QNRF_ECCV18"
    if nested.exists() and (nested / "Train").exists():
        return nested
    return dest_dir


def build_split_manifest(split_dir: Path, split_name: str, root_rel: Path) -> list[dict]:
    items = []
    jpg_files = sorted(list(split_dir.glob("*.jpg")))
    print(f"Found {len(jpg_files)} images in {split_dir}")

    for img_p in jpg_files:
        stem = img_p.stem
        mat_p = split_dir / f"{stem}_ann.mat"
        if not mat_p.exists():
            # some naming might use stem + '_ann.mat' or without 'img_'
            candidates = list(split_dir.glob(f"{stem}*.mat"))
            if candidates:
                mat_p = candidates[0]
            else:
                raise FileNotFoundError(f"Missing annotation for {img_p}")

        mat = scipy.io.loadmat(str(mat_p))
        ann_points = mat["annPoints"].tolist()

        rel_path = (root_rel / split_dir.name / img_p.name).as_posix()
        item = {
            "image": rel_path,
            "points": ann_points,
            "id": f"qnrf_{split_name}_{stem}",
        }
        items.append(item)

    return items


def generate_manifests() -> None:
    data_dir = Path("data")
    zip_path = data_dir / "archives" / "UCF-QNRF_ECCV18.zip"
    dest_dir = data_dir / "ucf_qnrf"

    qnrf_root = extract_if_needed(zip_path, dest_dir)
    train_dir = qnrf_root / "Train"
    test_dir = qnrf_root / "Test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing Train/ or Test/ in {qnrf_root}")

    root_rel = qnrf_root.relative_to(data_dir)

    train_items = build_split_manifest(train_dir, "train", root_rel)
    test_items = build_split_manifest(test_dir, "test", root_rel)
    all_items = train_items + test_items

    tr_out = data_dir / "qnrf_train.jsonl"
    te_out = data_dir / "qnrf_test.jsonl"
    all_out = data_dir / "qnrf_all.jsonl"

    with tr_out.open("w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(train_items)} train samples to {tr_out}")

    with te_out.open("w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(test_items)} test samples to {te_out}")

    with all_out.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(all_items)} total samples to {all_out}")


if __name__ == "__main__":
    generate_manifests()
