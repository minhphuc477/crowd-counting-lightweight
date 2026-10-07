"""Generate canonical .jsonl manifests for UCSD Pedestrian benchmark.

Reads all 200 frames from each clip in data/ucsd/video/vidf/ and matches
annotations from data/ucsd/gt/vidf/*_frame_full.mat.

Creates:
- data/ucsd_all.jsonl (all clips, 2000 standard benchmark frames)
- data/ucsd_train.jsonl (standard benchmark training split)
- data/ucsd_test.jsonl (standard benchmark test split)
"""
from __future__ import annotations

import json
from pathlib import Path
import scipy.io


def generate_manifests() -> None:
    data_dir = Path("data")
    ucsd_dir = data_dir / "ucsd"
    vid_root = ucsd_dir / "video" / "vidf"
    gt_root = ucsd_dir / "gt" / "vidf"

    if not vid_root.exists() or not gt_root.exists():
        raise FileNotFoundError("UCSD video or gt directory not found")

    # In UCSD benchmark (Chan et al., C-3 Framework):
    # Scene 1 clips: vidf1_33_000 to vidf1_33_028 (or standard peds1: 000-009)
    # The first 10 clips (000 to 009) correspond to the 2,000 frames benchmark.
    clip_dirs = sorted([d for d in vid_root.iterdir() if d.is_dir() and d.name.startswith("vidf1_33_")])
    if not clip_dirs:
        # Fallback to all vidf clips
        clip_dirs = sorted([d for d in vid_root.iterdir() if d.is_dir()])

    print(f"Found {len(clip_dirs)} video clip directories in UCSD.")

    all_items = []
    # Standard 2,000 frames benchmark uses the first 10 clips (000 to 009, 200 frames each = 2,000 frames)
    benchmark_clips = clip_dirs[:10]

    for clip_dir in benchmark_clips:
        clip_name = clip_dir.stem  # e.g. vidf1_33_000
        mat_path = gt_root / f"{clip_name}_frame_full.mat"
        if not mat_path.exists():
            print(f"Warning: GT not found for {clip_name}, skipping.")
            continue

        f_mat = scipy.io.loadmat(str(mat_path))
        f_frames = f_mat["fgt"][0, 0]["frame"][0]

        frame_files = sorted(list(clip_dir.glob("*.png")))
        for f_idx, frame_file in enumerate(frame_files):
            # 1-indexed frame
            frame_num = f_idx + 1
            if f_idx < len(f_frames):
                loc = f_frames[f_idx]["loc"][0, 0]
                # loc is Nx3: [x, y, weight]
                points = loc[:, :2].tolist() if loc.size > 0 else []
            else:
                points = []

            rel_img = f"ucsd/video/vidf/{clip_dir.name}/{frame_file.name}"
            item = {
                "image": rel_img,
                "points": points,
                "id": f"{clip_name}_f{frame_num:03d}",
            }
            all_items.append(item)

    print(f"Total UCSD items compiled: {len(all_items)}")

    # 1. Write ucsd_all.jsonl
    all_out = data_dir / "ucsd_all.jsonl"
    with all_out.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    print(f"Wrote {len(all_items)} samples to {all_out}")

    # Standard UCSD Peds1 split: 601-1400 (800 frames) train, remaining (1200 frames) test
    # (Frames 1-600 test, 601-1400 train, 1401-2000 test)
    if len(all_items) >= 2000:
        train_items = all_items[600:1400]
        test_items = all_items[:600] + all_items[1400:2000]

        tr_out = data_dir / "ucsd_train.jsonl"
        te_out = data_dir / "ucsd_test.jsonl"

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
