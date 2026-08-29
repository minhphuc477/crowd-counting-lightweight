"""Visualize the mass map with either local-max or OT-M point decoding.

Generates side-by-side comparisons:
- Left: Original Image + Ground Truth Point Annotations (Red)
- Middle: Predicted Stride-4 Density Heatmap Overlay
- Right: Parameter-free predicted locations (Green) + count comparison
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

_REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from hpc.models.hpc_lite import HPCLite
from hpc.data.sha import ShanghaiTechDataset
from hpc.metrics.localization import extract_points_from_mass_map
from hpc.metrics.otm import otm_localize
from hpc.models.factory import build_model_from_config


def apply_colormap_jet(density_norm: np.ndarray) -> np.ndarray:
    """Simple jet-like colormap in pure NumPy (values in 0..1 to RGB 0..255)."""
    # 4-stage color ramp: Blue -> Cyan -> Yellow -> Red
    r = np.clip(1.5 - np.abs(4.0 * density_norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * density_norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * density_norm - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def visualize_crowd_localization(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = "runs/sha/visualizations",
    num_samples: int = 5,
    method: str = "otm",
    seed: int = 42,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model = build_model_from_config(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded model from: {checkpoint_path}")

    # 2. Load Test Dataset
    d_cfg = cfg["dataset"]
    dataset = ShanghaiTechDataset(
        root=d_cfg["root"],
        part=d_cfg.get("part", "part_A"),
        split="test_data",
        is_train=False,
        crop_size=d_cfg["crop_size"],
        image_mean=d_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=d_cfg.get("image_std", [0.229, 0.224, 0.225]),
    )
    print(f"Loaded {len(dataset)} test samples. Visualizing {num_samples} samples...")

    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for idx in indices:
        sample = dataset[idx]
        img_path = sample["img_path"]
        gt_cnt = float(sample["gt_count"])
        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size

        # Model inference
        img_tensor = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_cnt, d_map = model.predict(img_tensor, pad_multiple=16)

        pred_val = float(pred_cnt.item())
        d_np = d_map.squeeze().cpu().numpy()  # (H/4, W/4)

        # 1. Ground Truth Canvas
        gt_canvas = raw_img.copy()
        draw_gt = ImageDraw.Draw(gt_canvas)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_pts = np.asarray(sample["gt_points"], dtype=np.float32).reshape(-1, 2)
        for pt in gt_pts:
            px, py = float(pt[0]), float(pt[1])
            draw_gt.ellipse([px - 3, py - 3, px + 3, py + 3], fill="red", outline="black")

        # 2. Density Heatmap Canvas
        d_rescaled = (d_np - d_np.min()) / (d_np.max() - d_np.min() + 1e-8)
        heatmap_np = apply_colormap_jet(d_rescaled)
        heatmap_img = Image.fromarray(heatmap_np).resize((w, h), Image.Resampling.BILINEAR)
        # Blend with original image
        blend_canvas = Image.blend(raw_img, heatmap_img, alpha=0.55)

        # 3. Predicted Localization Points Canvas
        pred_canvas = raw_img.copy()
        draw_pred = ImageDraw.Draw(pred_canvas)
        if method == "otm":
            pred_pts = otm_localize(
                d_map[0, 0], output_stride=4, image_hw=(h, w), seed=seed + int(idx)
            ).cpu().numpy()
        elif method == "local_max":
            pred_pts = extract_points_from_mass_map(d_np, stride=4)
        else:
            raise ValueError("method must be 'otm' or 'local_max'")
        for pt in pred_pts:
            px, py = float(pt[0]), float(pt[1])
            draw_pred.ellipse([px - 3, py - 3, px + 3, py + 3], fill="lime", outline="black")

        # Combine 3 panels horizontally
        combined = Image.new("RGB", (w * 3, h + 50), color=(25, 25, 25))
        combined.paste(gt_canvas, (0, 50))
        combined.paste(blend_canvas, (w, 50))
        combined.paste(pred_canvas, (w * 2, 50))

        # Title bar text
        draw_combined = ImageDraw.Draw(combined)
        title_gt = f"GT: {int(gt_cnt)} people (Red dots)"
        title_heat = f"Predicted Heatmap (Stride-4 Density Map)"
        title_pred = f"Pred: {pred_val:.1f} people ({method}: {len(pred_pts)} points)"
        
        draw_combined.text((20, 15), title_gt, fill="white")
        draw_combined.text((w + 20, 15), title_heat, fill="white")
        draw_combined.text((w * 2 + 20, 15), title_pred, fill="white")

        save_path = os.path.join(output_dir, f"vis_sample_{idx+1}_{stem}.jpg")
        combined.save(save_path, quality=92)
        print(f"Saved visualization to: {save_path} (GT: {int(gt_cnt)}, Pred: {pred_val:.1f})")

    print(f"\nAll visualizations generated successfully in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sha.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="runs/sha/best.pt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="runs/sha/visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test images to visualize")
    parser.add_argument("--method", choices=["local_max", "otm"], default="otm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    visualize_crowd_localization(
        args.config, args.checkpoint, args.output_dir, args.num_samples, args.method, args.seed
    )
