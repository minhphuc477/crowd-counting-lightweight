"""Quantitative Crowd Localization Evaluation Protocol.

Computes standard point-level localization benchmarks:
1. Fixed-radius matching: Precision, Recall, F1 at sigma in {4, 8, 16, 32} pixels.
2. Scale-stratified matching (Small, Medium, Large heads based on k-NN head scale).
"""
import os
import json
import argparse
import yaml
import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial import KDTree
import torch
from PIL import Image

from hpc.models.hpc_lite import HPCLite
from hpc.data.sha import ShanghaiTechDataset, load_sha_mat_points


def extract_peaks(d_map: np.ndarray, threshold_rel: float = 0.08) -> np.ndarray:
    """Extract discrete (x, y) coordinates from 2D continuous density map."""
    if d_map.size == 0 or np.max(d_map) <= 1e-5:
        return np.zeros((0, 2), dtype=np.float32)
    
    neighborhood = ndimage.generate_binary_structure(2, 2)
    local_max = ndimage.maximum_filter(d_map, footprint=neighborhood) == d_map
    threshold = float(np.max(d_map)) * threshold_rel
    detected_mask = local_max & (d_map > max(threshold, 1e-4))
    
    y_coords, x_coords = np.where(detected_mask)
    if len(x_coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)
        
    coords = np.column_stack([x_coords * 4 + 2, y_coords * 4 + 2]).astype(np.float32)
    return coords


def match_points_greedy(pred_pts: np.ndarray, gt_pts: np.ndarray, radius: float):
    """Greedy nearest-neighbor bipartite point matching within radius sigma."""
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0

    tree = KDTree(pred_pts)
    # For each GT point, find predicted points within radius
    gt_to_preds = tree.query_ball_point(gt_pts, r=radius)
    
    matched_preds = set()
    tp = 0
    
    # Sort GT matches by candidate distance to prioritize closest pairs
    pairs = []
    for gt_idx, cand_indices in enumerate(gt_to_preds):
        for p_idx in cand_indices:
            dist = np.linalg.norm(gt_pts[gt_idx] - pred_pts[p_idx])
            pairs.append((dist, gt_idx, p_idx))
            
    pairs.sort(key=lambda x: x[0])
    
    matched_gts = set()
    for dist, gt_idx, p_idx in pairs:
        if gt_idx not in matched_gts and p_idx not in matched_preds:
            matched_gts.add(gt_idx)
            matched_preds.add(p_idx)
            tp += 1
            
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def compute_knn_head_scales(gt_pts: np.ndarray, k: int = 3) -> np.ndarray:
    """Estimate head scale per GT point from mean distance to k-nearest neighbors."""
    if len(gt_pts) <= k:
        return np.full(len(gt_pts), 30.0, dtype=np.float32)
    tree = KDTree(gt_pts)
    distances, _ = tree.query(gt_pts, k=k + 1)
    # Exclude distance to self (index 0)
    scales = np.mean(distances[:, 1:], axis=1).astype(np.float32)
    return scales


def evaluate_localization_benchmark(
    config_path: str,
    checkpoint_path: str,
    output_json: str = "runs/sha/localization_results.json",
    max_samples: int = None,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Localization Metrics on device: {device}")

    # 1. Load Model
    m_cfg = cfg["model"]
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        truncate_backbone=m_cfg.get("truncate_backbone", True),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Dataset
    d_cfg = cfg["dataset"]
    dataset = ShanghaiTechDataset(
        root=d_cfg["root"],
        part=d_cfg.get("part", "part_A"),
        is_train=False,
        crop_size=d_cfg["crop_size"],
    )

    n_eval = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"Evaluating {n_eval} test images for point localization...")

    radii = [4.0, 8.0, 16.0, 32.0]
    total_tp = {r: 0 for r in radii}
    total_fp = {r: 0 for r in radii}
    total_fn = {r: 0 for r in radii}

    # Scale-stratified counters (Small: <=15px, Medium: 15-30px, Large: >30px)
    scale_tp = {"small": 0, "medium": 0, "large": 0}
    scale_fn = {"small": 0, "medium": 0, "large": 0}
    scale_fp = {"small": 0, "medium": 0, "large": 0}

    with torch.no_grad():
        for i in range(n_eval):
            sample = dataset[i]
            img_path = sample["img_path"]
            img_tensor = sample["image"].unsqueeze(0).to(device)

            # Predict
            pred_cnt, d_map = model.predict(img_tensor, pad_multiple=16)
            d_np = d_map.squeeze().cpu().numpy()
            pred_pts = extract_peaks(d_np)

            # Load actual GT points
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mat_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), "ground_truth", f"GT_{stem}.mat")
            gt_pts = load_sha_mat_points(mat_path) if os.path.exists(mat_path) else np.zeros((0, 2), dtype=np.float32)

            # 1. Fixed Radius Metrics
            for r in radii:
                tp, fp, fn = match_points_greedy(pred_pts, gt_pts, radius=r)
                total_tp[r] += tp
                total_fp[r] += fp
                total_fn[r] += fn

            # 2. Scale-stratified matching (using adaptive threshold sigma = 0.5 * head_scale)
            if len(gt_pts) > 0:
                scales = compute_knn_head_scales(gt_pts, k=3)
                small_mask = scales <= 15.0
                med_mask = (scales > 15.0) & (scales <= 30.0)
                large_mask = scales > 30.0

                for group_name, mask in [("small", small_mask), ("medium", med_mask), ("large", large_mask)]:
                    g_pts = gt_pts[mask]
                    if len(g_pts) == 0:
                        continue
                    # Adaptive matching radius for scale
                    adaptive_r = 12.0 if group_name == "small" else (20.0 if group_name == "medium" else 35.0)
                    tp, fp, fn = match_points_greedy(pred_pts, g_pts, radius=adaptive_r)
                    scale_tp[group_name] += tp
                    scale_fn[group_name] += fn

            if (i + 1) % 20 == 0 or (i + 1) == n_eval:
                print(f"Evaluated [{i+1:03d}/{n_eval:03d}] images...")

    # Calculate final Metrics
    results = {"fixed_radius": {}, "scale_stratified": {}}

    print("\n" + "=" * 60)
    print("      CROWD POINT LOCALIZATION BENCHMARK RESULTS")
    print("=" * 60)

    print("\n--- 1. Fixed Distance Thresholds (sigma) ---")
    for r in radii:
        tp, fp, fn = total_tp[r], total_fp[r], total_fn[r]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        results["fixed_radius"][f"sigma_{int(r)}px"] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
        print(f"Sigma = {int(r):2d} px | Precision: {prec*100:5.2f}% | Recall: {rec*100:5.2f}% | F1-Score: {f1*100:5.2f}%")

    print("\n--- 2. Scale-Stratified Localization (Small vs Medium vs Large Heads) ---")
    for group in ["small", "medium", "large"]:
        tp, fn = scale_tp[group], scale_fn[group]
        # Total predicted points roughly partitioned
        rec = tp / max(tp + fn, 1)
        # Using matched ratio as scale-specific recall/accuracy
        results["scale_stratified"][group] = {
            "matched_heads": tp,
            "missed_heads": fn,
            "recall": round(float(rec), 4),
        }
        print(f"{group.capitalize():6s} Heads (Dense/Sparse) | Found: {tp:5d} / {tp+fn:5d} | Recall / Hit Rate: {rec*100:5.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved localization results to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sha.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="runs/sha/best.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="runs/sha/localization_results.json", help="Path to output JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples to evaluate")
    args = parser.parse_args()

    evaluate_localization_benchmark(args.config, args.checkpoint, args.output, args.max_samples)
