"""Create a small, explicit synthetic ShanghaiTech-format smoke dataset."""

from __future__ import annotations

import argparse
import os
import numpy as np
from PIL import Image
import scipy.io as sio


def create_synthetic_crowd_dataset(
    root: str = "./data/synthetic_crowd",
    num_train: int = 16,
    num_val: int = 8,
    crop_size: int = 448,
):
    """Create synthetic dataset mimicking ShanghaiTech 1-based format."""
    if num_train <= 0 or num_val <= 0:
        raise ValueError("num_train and num_val must be positive")
    if crop_size < 64 or crop_size % 64 != 0:
        raise ValueError("crop_size must be at least 64 and divisible by 64 for NTPC")
    root = os.path.abspath(root)
    if os.path.isdir(root) and any(os.scandir(root)):
        raise FileExistsError(
            f"Refusing to overwrite non-empty smoke dataset directory: {root}"
        )
    for split, count in [("train_data", num_train), ("test_data", num_val)]:
        img_dir = os.path.join(root, "part_A", split, "images")
        gt_dir = os.path.join(root, "part_A", split, "ground-truth")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        np.random.seed(42 if split == "train_data" else 100)

        # Include empty scenes (0 points), sparse (1-10 points), medium (20-80), dense (200-600)
        point_distribution = [0, 0, 2, 5, 12, 25, 45, 80, 150, 250, 400, 600]

        for i in range(count):
            img_name = f"IMG_{i+1}.jpg"
            img_path = os.path.join(img_dir, img_name)
            mat_name = f"GT_IMG_{i+1}.mat"
            mat_path = os.path.join(gt_dir, mat_name)

            # Generate synthetic image
            img_array = np.random.randint(40, 220, size=(crop_size, crop_size, 3), dtype=np.uint8)
            img_pil = Image.fromarray(img_array)
            img_pil.save(img_path)

            num_pts = point_distribution[i % len(point_distribution)]
            if num_pts > 0:
                pts_0based = np.random.uniform(10, crop_size - 10, size=(num_pts, 2)).astype(np.float32)
                # Save as 1-based MATLAB coordinates [1, W] x [1, H]
                pts_1based = pts_0based + 1.0
            else:
                pts_1based = np.zeros((0, 2), dtype=np.float32)

            # Save ShanghaiTech format .mat
            image_info = np.zeros((1, 1), dtype=object)
            location_dict = np.zeros((1, 1), dtype=object)
            location_dict[0, 0] = (pts_1based,)
            image_info[0, 0] = location_dict

            sio.savemat(mat_path, {"image_info": image_info, "annPoints": pts_1based, "number": num_pts})

    print(f"Synthetic crowd dataset created at {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="./data/synthetic_crowd")
    parser.add_argument("--num-train", type=int, default=16)
    parser.add_argument("--num-test", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    args = parser.parse_args()
    create_synthetic_crowd_dataset(
        root=args.root,
        num_train=args.num_train,
        num_val=args.num_test,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
