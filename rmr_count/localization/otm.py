from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch

from rmr_count.localization.metrics import LocalizationMeter


def otm_density_to_points(
    density: torch.Tensor | np.ndarray,
    stride: int = 4,
    eps: float = 2.0,
    outer_iters: int = 8,
    sinkhorn_iters: int = 25,
    tau: float = 1e-4,
    seed: int | None = 42,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Parameter-free Optimal Transport Minimization (OT-M) converting density/count field to point set.

    Scientific Notice:
        This implementation is a pilot diagnostic Sinkhorn-based point extractor for L2/L3 mechanism
        validation. It implements alternating entropic optimal transport (Sinkhorn + barycentric projection).
        While inspired by Lin et al. ('Optimal Transport Minimization', CVPR 2023), it is an internal
        diagnostic tool and should not be cited as a faithful replication of the official CVPR 2023
        author codebase without formal calibration.

    Args:
        density: 2D array-like of shape (H_g, W_g) representing predicted count measure.
        stride: Spatial stride in pixels (default 4).
        eps: Entropic regularization parameter (default 2.0 px^2).
        outer_iters: Number of alternating OT-M updates (default 8).
        sinkhorn_iters: Number of Sinkhorn scaling iterations per outer step (default 25).
        tau: Minimum threshold fraction of max density to consider as active source mass.
        seed: Random seed for deterministic reproducibility in point sampling and jitter (default 42).
        device: PyTorch device ('cuda' or 'cpu'). Defaults to cuda if available.

    Returns:
        Point coordinates array of shape (m, 2) in [x, y], where m = round(sum(density)).
    """
    if isinstance(density, np.ndarray):
        density_t = torch.from_numpy(density).float()
    else:
        density_t = density.detach().float()

    if density_t.ndim > 2:
        density_t = density_t.squeeze()

    total_mass = float(density_t.sum().item())
    m = int(round(total_mass))
    if m <= 0:
        return np.empty((0, 2), dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    density_t = density_t.to(device)

    # Active source locations
    max_val = float(density_t.max().item())
    threshold = max(tau * max_val, 1e-7)
    active_mask = density_t > threshold
    if not active_mask.any():
        active_mask = density_t > 0
        if not active_mask.any():
            return np.empty((0, 2), dtype=np.float32)

    active_indices = torch.nonzero(active_mask, as_tuple=False)  # (K, 2) -> (row, col)
    a = density_t[active_mask].clone()
    k_src = a.numel()

    # Source spatial coordinates in image pixel space: x = s*col + s/2, y = s*row + s/2
    src_y = (active_indices[:, 0].float() * stride) + (stride / 2.0)
    src_x = (active_indices[:, 1].float() * stride) + (stride / 2.0)
    x_src = torch.stack([src_x, src_y], dim=-1)  # (K, 2)

    # Normalize source weights to sum to 1
    a = a / a.sum()

    # Generator for deterministic execution
    if seed is not None:
        gen = torch.Generator(device=device if device.type == "cpu" else "cpu")
        gen.manual_seed(seed)
    else:
        gen = None

    # Initialize target points B = {y_j}_{j=1}^m
    with torch.no_grad():
        if m <= k_src:
            _, topk_idx = torch.topk(a, k=m)
            y_pts = x_src[topk_idx].clone()
        else:
            # Deterministic multinomial sampling via CPU generator if needed
            a_cpu = a.cpu()
            if gen is not None:
                sampled_idx = torch.multinomial(a_cpu, num_samples=m, replacement=True, generator=gen).to(device)
            else:
                sampled_idx = torch.multinomial(a, num_samples=m, replacement=True)
            y_pts = x_src[sampled_idx].clone()

        # Add small Gaussian jitter (0.25 px std) to break point degeneracy in multi-occupancy cells
        if gen is not None:
            jitter_cpu = torch.randn(y_pts.shape, generator=gen, dtype=torch.float32) * (stride * 0.1)
            jitter = jitter_cpu.to(device)
        else:
            jitter = torch.randn_like(y_pts) * (stride * 0.1)
        y_pts = y_pts + jitter

        b = torch.full((m,), 1.0 / m, dtype=torch.float32, device=device)

        # Precompute x_src norm squared: ||x||^2 shape (K, 1)
        x2 = torch.sum(x_src ** 2, dim=-1, keepdim=True)

        # Alternating Sinkhorn + Barycentric update
        for _ in range(outer_iters):
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
            y2 = torch.sum(y_pts ** 2, dim=-1, keepdim=True).transpose(0, 1)  # (1, m)
            cost = x2 + y2 - 2.0 * torch.matmul(x_src, y_pts.transpose(0, 1))
            cost = torch.clamp(cost, min=0.0)

            # Numerical stability: shift cost matrix by min cost along rows
            min_cost = cost.min(dim=1, keepdim=True)[0]
            kernel = torch.exp(-(cost - min_cost) / eps)

            # Sinkhorn iterations
            v = torch.ones_like(b)
            for _ in range(sinkhorn_iters):
                u = a / (torch.matmul(kernel, v) + 1e-12)
                v = b / (torch.matmul(kernel.transpose(0, 1), u) + 1e-12)

            # Barycentric update: y_j = (sum_i T_ij * x_i) / (sum_i T_ij)
            u_k = u.unsqueeze(1) * kernel  # (K, m)
            transport_weights = u_k * v.unsqueeze(0)  # (K, m)
            col_sums = transport_weights.sum(dim=0, keepdim=True) + 1e-12  # (1, m)
            prob_cols = transport_weights / col_sums  # (K, m)

            # New coordinates: y_pts = prob_cols^T @ x_src
            y_pts = torch.matmul(prob_cols.transpose(0, 1), x_src)

    out_pts = y_pts.detach().cpu().numpy()
    del density_t, a, x_src, y_pts
    return out_pts


def evaluate_oracle_otm(
    manifest_path: str | Path,
    stride: int = 4,
    sigmas: list[float] | tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0),
    eps: float = 2.0,
    outer_iters: int = 8,
) -> dict[str, Any]:
    """Runs L2 Oracle OT-M evaluation on a dataset manifest using exact GT raster counts."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    meter = LocalizationMeter(sigmas=sigmas)
    num_images = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            num_images += 1

            img_path = Path(item["image"])
            if not img_path.is_absolute() and not img_path.exists():
                candidate = path.parent / img_path
                if candidate.exists():
                    img_path = candidate

            raw_points = item.get("points", [])
            points = np.asarray(raw_points, dtype=np.float32)
            if points.size == 0 or len(raw_points) == 0:
                points = np.empty((0, 2), dtype=np.float32)
            elif points.ndim == 1:
                points = points.reshape(-1, 2)

            # Image dimensions
            if img_path.exists():
                with Image.open(img_path) as img:
                    w, h = img.size
            else:
                if len(points) > 0:
                    w = int(math.ceil(np.max(points[:, 0]) + 1))
                    h = int(math.ceil(np.max(points[:, 1]) + 1))
                else:
                    w, h = 1024, 768

            gw = int(math.ceil(w / stride))
            gh = int(math.ceil(h / stride))

            if len(points) == 0:
                meter.update(np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32))
                continue

            x = points[:, 0]
            y = points[:, 1]
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            valid_gts = points[valid]

            if len(valid_gts) == 0:
                meter.update(np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32))
                continue

            # Rasterize GT to grid
            j = np.clip(np.floor((valid_gts[:, 0] + 0.5) / stride).astype(np.int64), 0, gw - 1)
            i = np.clip(np.floor((valid_gts[:, 1] + 0.5) / stride).astype(np.int64), 0, gh - 1)

            y_gt = np.zeros((gh, gw), dtype=np.float32)
            np.add.at(y_gt, (i, j), 1.0)

            # Run OT-M on ground-truth raster
            otm_pts = otm_density_to_points(
                y_gt,
                stride=stride,
                eps=eps,
                outer_iters=outer_iters,
                seed=42,
            )

            meter.update(otm_pts, valid_gts)
            if num_images % 20 == 0:
                gc.collect()

    summary = meter.compute_summary()
    summary["manifest"] = str(path.name)
    summary["num_images"] = num_images
    summary["stride"] = stride
    summary["eps"] = eps
    summary["outer_iters"] = outer_iters
    return summary


def evaluate_model_otm(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    stride: int = 4,
    sigmas: list[float] | tuple[float, ...] = (4.0, 8.0),
    device: str = "cuda",
) -> dict[str, Any]:
    """Runs L3 Predicted Model OT-M evaluation on a dataset manifest using trained weights."""
    from rmr_count.eval import make_model_from_ckpt, predict_tiled
    from torchvision import transforms

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_t = torch.device(device if (torch.cuda.is_available() or device != "cuda") else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device_t)
    model = make_model_from_ckpt(ckpt, device=device_t)
    model.to(device_t)
    model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    path = Path(manifest_path)
    meter = LocalizationMeter(sigmas=sigmas)
    num_images = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            num_images += 1

            img_path = Path(item["image"])
            if not img_path.is_absolute() and not img_path.exists():
                candidate = path.parent / img_path
                if candidate.exists():
                    img_path = candidate

            raw_points = item.get("points", [])
            points = np.asarray(raw_points, dtype=np.float32)
            if points.size == 0 or len(raw_points) == 0:
                points = np.empty((0, 2), dtype=np.float32)
            elif points.ndim == 1:
                points = points.reshape(-1, 2)

            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                w, h = img_rgb.size
                tensor = tf(img_rgb).to(device_t)

            if len(points) == 0:
                valid_gts = np.empty((0, 2), dtype=np.float32)
            else:
                valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
                valid_gts = points[valid]

            with torch.inference_mode():
                y_canvas = predict_tiled(model, tensor, tile_size=512, halo=0)
                y_pred = y_canvas[0].cpu().numpy()

            otm_pts = otm_density_to_points(y_pred, stride=stride, device=device_t, seed=42)
            meter.update(otm_pts, valid_gts)
            if num_images % 20 == 0:
                gc.collect()

    summary = meter.compute_summary()
    summary["manifest"] = str(path.name)
    summary["num_images"] = num_images
    summary["checkpoint"] = str(ckpt_path.name)
    summary["stride"] = stride
    return summary


def format_otm_table(summary: dict[str, Any], title: str = "OT-M Results") -> str:
    """Formats summary into a clean markdown table."""
    lines = [
        f"### {title} ({summary.get('manifest', '')}, images={summary['num_images']}, points={summary['total_ground_truth']:,})\n",
        "| Distance Threshold $\\sigma$ | Micro Precision | Micro Recall | Micro $F_1$ | Macro Precision | Macro Recall | Macro $F_1$ | TP | FP | FN |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, data in summary["thresholds"].items():
        s = data["sigma"]
        lines.append(
            f"| **$\\sigma = {int(s) if s.is_integer() else s:.1f}$ px** | "
            f"{data['micro_precision']*100:.2f}% | {data['micro_recall']*100:.2f}% | **{data['micro_f1']*100:.2f}%** | "
            f"{data['macro_precision']*100:.2f}% | {data['macro_recall']*100:.2f}% | {data['macro_f1']*100:.2f}% | "
            f"{data['tp']:,} | {data['fp']:,} | {data['fn']:,} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2/L3: Optimal Transport Minimization (OT-M) evaluator.")
    parser.add_argument("--mode", choices=["oracle", "model"], default="oracle", help="Evaluation mode.")
    parser.add_argument("--manifest", default="data/sha_a_test.jsonl", help="JSONL manifest path.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path (for mode=model).")
    parser.add_argument("--stride", type=int, default=4, help="Grid stride (default 4).")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[2.0, 4.0, 6.0, 8.0, 10.0], help="Sigmas.")
    parser.add_argument("--eps", type=float, default=2.0, help="Entropic regularization (default 2.0).")
    parser.add_argument("--outer-iters", type=int, default=8, help="Number of OT-M outer iterations (default 8).")
    parser.add_argument("--device", default="cuda", help="Computation device ('cuda' or 'cpu').")
    args = parser.parse_args()

    if args.mode == "oracle":
        print(f"=== Running L2 Oracle OT-M on {args.manifest} (stride={args.stride}, eps={args.eps}, iters={args.outer_iters}) ===")
        summary = evaluate_oracle_otm(
            args.manifest,
            stride=args.stride,
            sigmas=args.sigmas,
            eps=args.eps,
            outer_iters=args.outer_iters,
        )
        print("\n" + format_otm_table(summary, title=f"L2 Oracle OT-M (stride={args.stride})"))
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode model")
        print(f"=== Running L3 Model OT-M on {args.manifest} with {args.checkpoint} ===")
        summary = evaluate_model_otm(
            args.checkpoint,
            args.manifest,
            stride=args.stride,
            sigmas=args.sigmas,
            device=args.device,
        )
        print("\n" + format_otm_table(summary, title=f"L3 Model OT-M ({args.checkpoint})"))


if __name__ == "__main__":
    main()
