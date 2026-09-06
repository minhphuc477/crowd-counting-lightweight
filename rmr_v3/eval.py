from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_count.data import CrowdManifestDataset, collate_eval
from rmr_count.metrics import game_single, summarize_predictions

from .diagnostics import (
    compute_reliability_correlations,
    regional_reliability_rows,
)
from .model import RMRv3, RMRv3Config


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[RMRv3, bool, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    output_stride = int(m_cfg.get("output_stride", 4))
    feature_width = int(m_cfg.get("feature_width", 32))
    backbone_name = str(m_cfg.get("backbone", m_cfg.get("backbone_name", "mobilenetv4_conv_small_050.e3000_r224_in1k")))
    pretrained = False  # loading weights from ckpt

    init_m0 = float(m_cfg.get("init_m0", 0.015763))
    region_sizes_px = tuple(int(x) for x in m_cfg.get("region_sizes_px", (32, 64, 128)))
    region_overlap = float(m_cfg.get("region_overlap", 0.5))
    include_full_image = bool(m_cfg.get("include_full_image", False))

    iterations = int(m_cfg.get("iterations", 2))
    omega = float(m_cfg.get("omega", m_cfg.get("sirt_omega", 1.0)))
    residual_clip = float(m_cfg.get("residual_clip", 0.0))

    dispersion_init = float(m_cfg.get("dispersion_init", 50.0))
    dispersion_min = float(m_cfg.get("dispersion_min", 0.5))
    dispersion_max = float(m_cfg.get("dispersion_max", 500.0))

    reliability_mode = str(m_cfg.get("reliability_mode", "nb_rate_variance"))
    reliability_rate_std_floor = float(m_cfg.get("reliability_rate_std_floor", 0.01))
    reliability_weight_min = float(m_cfg.get("reliability_weight_min", 0.25))
    reliability_weight_max = float(m_cfg.get("reliability_weight_max", 4.0))
    normalize_reliability_within_scale = bool(m_cfg.get("normalize_reliability_within_scale", True))

    detach_region_mean_in_solver = bool(m_cfg.get("detach_region_mean_in_solver", True))
    detach_reliability_in_solver = bool(m_cfg.get("detach_reliability_in_solver", True))

    uniform_reliability = bool(m_cfg.get("uniform_reliability", False))

    config = RMRv3Config(
        output_stride=output_stride,
        feature_width=feature_width,
        backbone_name=backbone_name,
        pretrained=pretrained,
        init_m0=init_m0,
        region_sizes_px=region_sizes_px,
        region_overlap=region_overlap,
        include_full_image=include_full_image,
        iterations=iterations,
        omega=omega,
        residual_clip=residual_clip,
        dispersion_init=dispersion_init,
        dispersion_min=dispersion_min,
        dispersion_max=dispersion_max,
        reliability_mode=reliability_mode,
        reliability_rate_std_floor=reliability_rate_std_floor,
        reliability_weight_min=reliability_weight_min,
        reliability_weight_max=reliability_weight_max,
        normalize_reliability_within_scale=normalize_reliability_within_scale,
        detach_region_mean_in_solver=detach_region_mean_in_solver,
        detach_reliability_in_solver=detach_reliability_in_solver,
    )

    model = RMRv3(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, uniform_reliability, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RMR-v3 checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--manifest", default=None, help="Path to eval manifest jsonl")
    ap.add_argument("--output-dir", default=None, help="Directory to save evaluation artifacts")
    ap.add_argument("--uniform-reliability", action="store_true", default=None, help="Override uniform reliability setting")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, ckpt_uniform, cfg = load_model_from_ckpt(ckpt_path, device)
    uniform_reliability = ckpt_uniform if args.uniform_reliability is None else args.uniform_reliability

    manifest = args.manifest or cfg.get("data", {}).get("val_manifest", "data/sha_a_val.jsonl")
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval_val"
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = int(cfg.get("model", {}).get("output_stride", 4))
    dataset = CrowdManifestDataset(manifest, train=False, output_stride=stride)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    pred_rows = []
    diag_rows = []

    print(f"Evaluating {ckpt_path.name} on {manifest} ({len(dataset)} samples)...", flush=True)

    with torch.no_grad():
        for i, batch_list in enumerate(loader):
            for sample in batch_list:
                image = sample["image"].unsqueeze(0).to(device)
                target = sample["target_y"].to(device)

                out = model(image, uniform_reliability=uniform_reliability)
                y = out["y"][0]
                pred = float(y.sum().item())
                gt = float(target.sum().item())

                row = {
                    "index": i,
                    "gt": gt,
                    "pred": pred,
                    "abs_err": abs(pred - gt),
                    "sq_err": (pred - gt) ** 2,
                }
                for level in range(4):
                    row[f"GAME{level}"] = game_single(y, target, level)
                pred_rows.append(row)

                d_rows = regional_reliability_rows(out, target.unsqueeze(0))
                for r in d_rows:
                    r["sample_index"] = i
                diag_rows.extend(d_rows)

    summary = summarize_predictions(pred_rows)
    corrs = compute_reliability_correlations(diag_rows)
    summary.update(corrs)

    # Density-stratified metrics with frozen thresholds [100, 500]
    gts = np.array([r["gt"] for r in pred_rows])
    aes = np.array([r["abs_err"] for r in pred_rows])

    sparse_mask = gts <= 100.0
    mod_mask = (gts > 100.0) & (gts <= 500.0)
    dense_mask = gts > 500.0

    summary["mae_sparse"] = float(np.mean(aes[sparse_mask])) if np.any(sparse_mask) else 0.0
    summary["mae_moderate"] = float(np.mean(aes[mod_mask])) if np.any(mod_mask) else 0.0
    summary["mae_dense"] = float(np.mean(aes[dense_mask])) if np.any(dense_mask) else 0.0

    summary["n_sparse"] = int(np.sum(sparse_mask))
    summary["n_moderate"] = int(np.sum(mod_mask))
    summary["n_dense"] = int(np.sum(dense_mask))

    # Weight distribution
    weights = np.array([r["weight"] for r in diag_rows])
    summary["weight_mean"] = float(np.mean(weights))
    summary["weight_std"] = float(np.std(weights))
    summary["weight_min"] = float(np.min(weights))
    summary["weight_max"] = float(np.max(weights))

    # Save summary.json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save predictions.csv
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    # Save reliability_diagnostics.csv
    if diag_rows:
        with open(out_dir / "reliability_diagnostics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)

    print(f"\nEvaluation Results:")
    print(f"  MAE: {summary['mae']:.2f} | RMSE: {summary['rmse']:.2f} | NAE: {summary['nae']:.3f} | Bias: {summary['bias']:+.2f}")
    print(f"  GAME0: {summary['GAME0']:.2f} | GAME1: {summary['GAME1']:.2f} | GAME2: {summary['GAME2']:.2f} | GAME3: {summary['GAME3']:.2f}")
    print(f"  Sparse MAE (<=100): {summary['mae_sparse']:.2f} (n={summary['n_sparse']})")
    print(f"  Moderate MAE (101-500): {summary['mae_moderate']:.2f} (n={summary['n_moderate']})")
    print(f"  Dense MAE (>500): {summary['mae_dense']:.2f} (n={summary['n_dense']})")
    print(f"  Pearson(var, err): {summary['pearson_rate_var_error']:.4f}")
    print(f"  Spearman(var, err): {summary['spearman_rate_var_error']:.4f}")
    print(f"  Spearman(weight, err): {summary['spearman_weight_error']:.4f}")
    print(f"Saved artifacts to {out_dir}\n")


if __name__ == "__main__":
    main()
