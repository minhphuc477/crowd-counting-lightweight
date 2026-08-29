"""Standalone evaluation tool for Crowd Counting & Point Localization on benchmark test sets.

Computes:
  - Global Counting Metrics: MAE, RMSE, NAE, Bias
  - Density Subgroup Diagnostics: Sparse, Medium, Dense MAE
  - Localization Metrics: Precision, Recall, F1-Score @ sigma in {4, 8, 16, 32} pixels
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

from hpc.data.sha import ShanghaiTechDataset
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.localization import evaluate_dataset_localization, extract_points_from_mass_map
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from hpc.models.hpc_lite import HPCLite


def main():
    parser = argparse.ArgumentParser(description="Evaluate Counting and Localization performance.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech", help="Dataset root path")
    parser.add_argument("--part", type=str, default="part_A", help="Dataset part (part_A / part_B)")
    parser.add_argument("--stride", type=int, default=4, help="Mass map output stride (default 4)")
    parser.add_argument("--min_distance_px", type=int, default=4, help="Min peak distance in pixels")
    parser.add_argument("--threshold_abs", type=float, default=0.01, help="Min peak absolute value")
    parser.add_argument("--threshold_rel", type=float, default=0.05, help="Min peak relative value")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    m_cfg = cfg.get("model", {})
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_p8_context=bool(m_cfg.get("use_p8_context", True)),
        use_repblock=bool(m_cfg.get("use_repblock", False)),
        eps_d=float(m_cfg.get("eps_d", 1e-6)),
        truncate_backbone=bool(m_cfg.get("truncate_backbone", True)),
    ).to(device)

    model_state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    ds = ShanghaiTechDataset(
        root=args.data_root,
        part=args.part,
        split="test_data",
        is_train=False,
        image_mean=cfg.get("dataset", {}).get("image_mean", [0.5, 0.5, 0.5]),
        image_std=cfg.get("dataset", {}).get("image_std", [0.5, 0.5, 0.5]),
    )

    print(f"Evaluating {len(ds)} test samples for Counting & Localization...")

    predictions_cnt = []
    ground_truths_cnt = []
    predictions_pts = []
    ground_truths_pts = []

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            gt_cnt = float(sample["gt_count"])
            gt_pts = sample.get("points", np.empty((0, 2), dtype=np.float32))

            # Full image padded inference
            pred_cnt, pred_mass = model.predict(img, pad_multiple=32)
            
            # Extract point coordinates via local maxima on mass map D
            pred_pts = extract_points_from_mass_map(
                pred_mass[0, 0],
                stride=args.stride,
                threshold_rel=args.threshold_rel,
                threshold_abs=args.threshold_abs,
                min_distance_px=args.min_distance_px,
            )

            predictions_cnt.append(float(pred_cnt.item()))
            ground_truths_cnt.append(gt_cnt)
            predictions_pts.append(pred_pts)
            ground_truths_pts.append(gt_pts)

    counting_res = evaluate_counting_metrics(predictions_cnt, ground_truths_cnt)
    subgroup_res = evaluate_subgroup_diagnostics(predictions_cnt, ground_truths_cnt)
    loc_res = evaluate_dataset_localization(predictions_pts, ground_truths_pts, distance_thresholds=(4.0, 8.0, 16.0, 32.0))

    print("\n=======================================================")
    print(" COUNTING METRICS")
    print("=======================================================")
    print(f" MAE  : {counting_res['mae']:.2f}")
    print(f" RMSE : {counting_res['rmse']:.2f}")
    print(f" Bias : {counting_res['bias']:.2f}")
    print(f" NAE  : {counting_res['nae']:.4f}")
    print(f" Sparse [11-100] MAE   : {subgroup_res['bin_11_100_mae']:.2f}")
    print(f" Med [101-1000] MAE    : {subgroup_res['bin_101_1000_mae']:.2f}")
    print(f" Dense [>1000] MAE     : {subgroup_res['bin_gt1000_mae']:.2f}")

    print("\n=======================================================")
    print(" LOCALIZATION METRICS (Point-Matching Bipartite)")
    print("=======================================================")
    for sigma in (4, 8, 16, 32):
        prec = loc_res[f"sigma_{sigma}_precision"] * 100
        rec = loc_res[f"sigma_{sigma}_recall"] * 100
        f1 = loc_res[f"sigma_{sigma}_f1"] * 100
        tp = loc_res[f"sigma_{sigma}_tp"]
        fp = loc_res[f"sigma_{sigma}_fp"]
        fn = loc_res[f"sigma_{sigma}_fn"]
        print(f" Sigma = {sigma:2d} px | F1: {f1:5.2f}% | Prec: {prec:5.2f}% | Rec: {rec:5.2f}% (TP={tp}, FP={fp}, FN={fn})")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
