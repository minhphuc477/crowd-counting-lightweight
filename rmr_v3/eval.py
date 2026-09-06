from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset, save_evaluation_artifacts
from .diagnostics import (
    compute_dispersion_saturation,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
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
    model.set_solver_strength(1.0)
    model.to(device).eval()
    return model, uniform_reliability, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RMR-v3 checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--manifest", default=None, help="Path to eval manifest jsonl")
    ap.add_argument("--output-dir", default=None, help="Directory to save evaluation artifacts")
    ap.add_argument("--uniform-reliability", action="store_true", default=None, help="Override uniform reliability setting")
    ap.add_argument("--tiling", dest="tiling", action="store_true", default=True, help="Enable tiled prediction (default: True)")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false", help="Disable tiled prediction")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, ckpt_uniform, cfg = load_model_from_ckpt(ckpt_path, device)
    uniform_reliability = ckpt_uniform if args.uniform_reliability is None else args.uniform_reliability

    manifest = args.manifest or cfg.get("data", {}).get("val_manifest", "data/sha_a_val.jsonl")
    manifest_path = Path(manifest)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / f"eval_{manifest_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = int(cfg.get("model", {}).get("output_stride", 4))
    dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=stride)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    diag_rows: list[dict] = []
    traj_rows: list[dict] = []

    def sample_callback(sample: dict, out: dict, y: torch.Tensor, row: dict) -> dict:
        target = sample["target_y"].to(device)
        d_rows = regional_reliability_rows(out, target.unsqueeze(0))
        for r in d_rows:
            r["sample_index"] = row["index"]
            r["sample_id"] = row["id"]
        diag_rows.extend(d_rows)

        t_diag = compute_solver_trajectory_diagnostics(out, target.unsqueeze(0))
        if t_diag:
            traj_rows.append(t_diag)
        return {}

    print(f"Evaluating {ckpt_path.name} on {manifest_path} ({len(dataset)} samples)...", flush=True)

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=stride,
        run_tiling=args.tiling,
        forward_kwargs={"uniform_reliability": uniform_reliability},
        extra_sample_callback=sample_callback,
    )

    corrs = compute_reliability_correlations(diag_rows)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(diag_rows)
    summary["calibration"] = calib

    sat = compute_dispersion_saturation(diag_rows)
    summary.update(sat)

    if traj_rows:
        traj_summary: dict[str, float] = {}
        for k in traj_rows[0].keys():
            vals = [tr[k] for tr in traj_rows if k in tr]
            traj_summary[k] = float(np.mean(vals)) if vals else 0.0
        summary["solver_trajectory"] = traj_summary
        summary["solver_help_fraction"] = traj_summary.get("solver_help_fraction", 0.0)
        summary["solver_harm_fraction"] = traj_summary.get("solver_harm_fraction", 0.0)
        summary["energy_monotonic_fraction"] = traj_summary.get("energy_monotonic_fraction", 1.0)
        for t in range(3):
            if f"mae_reg_y{t}" in traj_summary:
                summary[f"mae_reg_y{t}"] = traj_summary[f"mae_reg_y{t}"]

    # Weight distribution statistics
    weights = np.array([r["weight"] for r in diag_rows]) if diag_rows else np.array([1.0])
    solver_weights = np.array([r.get("solver_weight", r["weight"]) for r in diag_rows]) if diag_rows else np.array([1.0])

    summary["weight_mean"] = float(np.mean(weights))
    summary["weight_std"] = float(np.std(weights))
    summary["weight_min"] = float(np.min(weights))
    summary["weight_max"] = float(np.max(weights))
    summary["solver_weight_mean"] = float(np.mean(solver_weights))
    summary["solver_weight_std"] = float(np.std(solver_weights))

    save_evaluation_artifacts(out_dir, rows, summary)

    # Save reliability_diagnostics.csv
    if diag_rows:
        with open(out_dir / "reliability_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)

    print(f"\nEvaluation Results:")
    print(f"  MAE: {summary['MAE']:.2f} | RMSE: {summary['RMSE']:.2f} | NAE: {summary['NAE']:.3f} | Bias: {summary['Bias']:+.2f}")
    print(f"  GAME0: {summary['GAME0']:.2f} | GAME1: {summary['GAME1']:.2f} | GAME2: {summary['GAME2']:.2f} | GAME3: {summary['GAME3']:.2f}")
    print(f"  Sparse MAE (<=100): {summary['mae_sparse']:.2f} (n={summary['n_sparse']})")
    print(f"  Moderate MAE (101-500): {summary['mae_moderate']:.2f} (n={summary['n_moderate']})")
    print(f"  Dense MAE (>500): {summary['mae_dense']:.2f} (n={summary['n_dense']})")
    if "pearson_rate_var_error" in summary:
        print(f"  Pearson(var, err): {summary['pearson_rate_var_error']:.4f}")
        print(f"  Spearman(var, err): {summary['spearman_rate_var_error']:.4f}")
        print(f"  Spearman(weight, err): {summary['spearman_weight_error']:.4f}")
    print(f"Saved artifacts to {out_dir}\n")


if __name__ == "__main__":
    main()
