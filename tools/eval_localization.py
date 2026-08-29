"""Benchmark Evaluation Tool for Joint Counting & OT-M Localization on Test Sets.

Computes:
  - Counting: MAE, RMSE, NAE, Bias, Density Subgroups (Sparse, Med, Dense)
  - Parameter-Free OT-M Localization: F1, Precision, Recall @ sigma in {4, 8, 16} px
  - Local Maxima Peak Finding (Baseline Comparison)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from hpc.data.sha import ShanghaiTechDataset
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.localization import evaluate_dataset_localization, match_points
from hpc.metrics.otm import otm_localize
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from hpc.models.hpc_lite import HPCLite


def main():
    parser = argparse.ArgumentParser(description="Evaluate NTPC Counting and OT-M Localization.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data_root", type=str, default="./data/ShanghaiTech", help="Dataset root path")
    parser.add_argument("--part", type=str, default="part_A", help="Dataset part (part_A / part_B)")
    parser.add_argument("--stride", type=int, default=4, help="Output stride (default 4)")
    parser.add_argument("--ot_epsilon", type=float, default=0.02, help="OT Sinkhorn epsilon")
    parser.add_argument("--ot_iters", type=int, default=50, help="OT Sinkhorn iterations")
    parser.add_argument("--outer_iters", type=int, default=8, help="OT-M outer iterations")
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

    print(f"Evaluating {len(ds)} test samples for Counting & OT-M Localization...")

    predictions_cnt = []
    ground_truths_cnt = []
    otm_points_list = []
    ground_truths_pts = []

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            gt_cnt = float(sample["gt_count"])
            gt_pts = sample.get("points", np.empty((0, 2), dtype=np.float32))

            pred_cnt, pred_mass = model.predict(img, pad_multiple=32)

            # OT-M parameter-free point localization from single mass map D
            otm_pts = otm_localize(
                pred_mass[0, 0],
                output_stride=args.stride,
                outer_iterations=args.outer_iters,
                sinkhorn_iterations=args.ot_iters,
                epsilon=args.ot_epsilon,
            )

            predictions_cnt.append(float(pred_cnt.item()))
            ground_truths_cnt.append(gt_cnt)
            otm_points_list.append(otm_pts.cpu().numpy())
            ground_truths_pts.append(gt_pts)

    counting_res = evaluate_counting_metrics(predictions_cnt, ground_truths_cnt)
    subgroup_res = evaluate_subgroup_diagnostics(predictions_cnt, ground_truths_cnt)
    loc_res = evaluate_dataset_localization(otm_points_list, ground_truths_pts, distance_thresholds=(4.0, 8.0, 16.0))

    print("\n=======================================================")
    print(" COUNTING METRICS")
    print("=======================================================")
    print(f" MAE  : {counting_res['mae']:.2f}")
    print(f" RMSE : {counting_res['rmse']:.2f}")
    print(f" Bias : {counting_res['bias']:.2f}")
    print(f" NAE  : {counting_res['nae']:.4f}")
    print(f" Sparse [11-100] MAE   : {subgroup_res.get('bin_11_100_mae', 0.0):.2f}")
    print(f" Med [101-1000] MAE    : {subgroup_res.get('bin_101_1000_mae', 0.0):.2f}")
    print(f" Dense [>1000] MAE     : {subgroup_res.get('bin_gt1000_mae', 0.0):.2f}")

    print("\n=======================================================")
    print(" OT-M LOCALIZATION METRICS (Hungarian Matching)")
    print("=======================================================")
    for sigma in (4, 8, 16):
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
