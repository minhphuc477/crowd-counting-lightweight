from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset, save_evaluation_artifacts
from .diagnostics import (
    compute_dispersion_saturation,
    compute_nb_interval_coverage,
    compute_reliability_correlations,
    compute_solver_trajectory_diagnostics,
    compute_uncertainty_calibration_bins,
    regional_reliability_rows,
)
from .model import RMRv3, RMRv3Config


def load_model_from_ckpt(
    ckpt_path: Path, device: torch.device, use_ema: bool = True
) -> tuple[RMRv3, bool, dict, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
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
    eps = float(m_cfg.get("eps", 1e-6))

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
        eps=eps,
        native_scale_pooling=bool(m_cfg.get("native_scale_pooling", False)),
        regional_feature_stats=str(m_cfg.get("regional_feature_stats", "mean")),
        neck_type=str(m_cfg.get("neck_type", "additive")),
        context_dilations=tuple(int(x) for x in m_cfg.get("context_dilations", (1, 2, 3))),
        use_aspp_gap=bool(m_cfg.get("use_aspp_gap", False)),
        aspp_dilations=tuple(int(x) for x in m_cfg.get("aspp_dilations", (1, 3, 6))),
        region_head_hidden=int(m_cfg.get("region_head_hidden", 48)),
        enable_solver=bool(m_cfg.get("enable_solver", True)),
        # RMR-v7 fields
        hurdle_head=bool(m_cfg.get("hurdle_head", False)),
        tv_lambda=float(m_cfg.get("tv_lambda", 0.0)),
        temp_softplus=bool(m_cfg.get("temp_softplus", False)),
        # ema_decay not needed at eval time (no training)
    )

    model = RMRv3(config)

    # Prefer EMA weights if saved (higher quality than live weights)
    if use_ema and "ema_model" in ckpt:
        # EMA state is float32; cast to model dtype (may be fp16 on GPU)
        ema_sd = ckpt["ema_model"]
        live_sd = model.state_dict()
        cast_sd = {k: ema_sd[k].to(dtype=live_sd[k].dtype) for k in live_sd if k in ema_sd}
        # Fill any missing keys from live checkpoint (shouldn't happen)
        for k in live_sd:
            if k not in cast_sd:
                cast_sd[k] = ckpt["model"][k]
        model.load_state_dict(cast_sd)
        print(f"Loaded EMA weights from checkpoint (ema_model key found).")
    else:
        model.load_state_dict(ckpt["model"])
        if not use_ema and "ema_model" in ckpt:
            print(f"Loaded live weights from checkpoint (--use-live-weights specified).")

    model.switch_to_deploy()
    model.set_solver_strength(1.0)
    model.to(device).eval()
    return model, uniform_reliability, cfg, ckpt


def compute_file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
    except Exception:
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        dirty = bool(status)
    except Exception:
        dirty = False
    return commit, dirty


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RMR-v3 checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--manifest", default=None, help="Path to eval manifest jsonl")
    ap.add_argument("--output-dir", default=None, help="Directory to save evaluation artifacts")
    ap.add_argument("--uniform-reliability", dest="uniform_reliability", action="store_true", default=None, help="Force uniform reliability (W=I)")
    ap.add_argument("--weighted-reliability", dest="uniform_reliability", action="store_false", help="Force weighted reliability (W=diag(w_R))")
    ap.add_argument("--tiling", dest="tiling", action="store_true", default=True, help="Enable tiled prediction (default: True)")
    ap.add_argument("--no-tiling", dest="tiling", action="store_false", help="Disable tiled prediction")
    ap.add_argument("--use-live-weights", dest="use_ema", action="store_false", default=True, help="Evaluate live checkpoint weights instead of EMA weights")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, ckpt_uniform, cfg, ckpt = load_model_from_ckpt(ckpt_path, device, use_ema=args.use_ema)
    uniform_reliability = ckpt_uniform if args.uniform_reliability is None else args.uniform_reliability

    manifest = args.manifest or cfg.get("data", {}).get("val_manifest", "data/sha_a_test.jsonl")
    manifest_path = Path(manifest)
    mode_tag = "uniform" if uniform_reliability else "weighted"
    tiling_tag = "" if args.tiling else "_notiling"
    default_dir_name = f"eval_{manifest_path.stem}_{mode_tag}{tiling_tag}"
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / default_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stride = int(cfg.get("model", {}).get("output_stride", 4))
    dataset = CrowdManifestDataset(
        manifest_path,
        train=False,
        output_stride=stride,
        data_root=cfg.get("data", {}).get("data_root"),
    )
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

        extra = {}
        if "iterates" in out and len(out["iterates"]) >= 3:
            gt = float(row["gt"])
            pred_y0 = float(out["iterates"][0].sum().item())
            pred_y1 = float(out["iterates"][1].sum().item())
            pred_y2 = float(out["iterates"][2].sum().item())
            extra["pred_y0"] = pred_y0
            extra["pred_y1"] = pred_y1
            extra["pred_y2"] = pred_y2
            extra["ae_y0"] = abs(pred_y0 - gt)
            extra["ae_y1"] = abs(pred_y1 - gt)
            extra["ae_y2"] = abs(pred_y2 - gt)
        return extra

    print(f"Evaluating {ckpt_path.name} on {manifest_path} ({len(dataset)} samples)...", flush=True)

    density_bins = tuple(float(x) for x in cfg.get("eval", {}).get("density_bins", [100.0, 500.0]))

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=stride,
        run_tiling=args.tiling,
        forward_kwargs={"uniform_reliability": uniform_reliability},
        extra_sample_callback=sample_callback,
        enforce_gt_consistency=True,
        density_bins=density_bins,
    )

    corrs = compute_reliability_correlations(diag_rows)
    summary.update(corrs)

    calib = compute_uncertainty_calibration_bins(diag_rows)
    summary["calibration"] = calib

    disp_min = float(getattr(model.cfg, "dispersion_min", cfg.get("model", {}).get("dispersion_min", 0.5)))
    disp_max = float(getattr(model.cfg, "dispersion_max", cfg.get("model", {}).get("dispersion_max", 500.0)))
    sat = compute_dispersion_saturation(diag_rows, disp_min=disp_min, disp_max=disp_max)
    summary.update(sat)

    nb_cov = compute_nb_interval_coverage(diag_rows)
    summary["nb_interval_coverage"] = nb_cov
    summary.update(nb_cov)

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

    if rows and "ae_y0" in rows[0]:
        summary["mae_y0"] = float(np.mean([r["ae_y0"] for r in rows]))
        summary["mae_y1"] = float(np.mean([r["ae_y1"] for r in rows]))
        summary["mae_y2"] = float(np.mean([r["ae_y2"] for r in rows]))

    # Weight distribution statistics
    weights = np.array([r["weight"] for r in diag_rows]) if diag_rows else np.array([1.0])
    solver_weights = np.array([r.get("solver_weight", r["weight"]) for r in diag_rows]) if diag_rows else np.array([1.0])

    summary["weight_mean"] = float(np.mean(weights))
    summary["weight_std"] = float(np.std(weights))
    summary["weight_min"] = float(np.min(weights))
    summary["weight_max"] = float(np.max(weights))
    summary["solver_weight_mean"] = float(np.mean(solver_weights))
    summary["solver_weight_std"] = float(np.std(solver_weights))
    eval_commit, eval_dirty = get_git_info()
    resolved_cfg_file = ckpt_path.parent / "resolved_config.yaml"
    if resolved_cfg_file.exists():
        cfg_sha = compute_file_sha256(resolved_cfg_file)
    else:
        cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

    summary["provenance"] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_commit": eval_commit,
        "git_dirty": eval_dirty,
        "training_commit": str(ckpt.get("git_commit", ckpt.get("provenance", {}).get("git_commit", "unknown"))),
        "training_git_dirty": ckpt.get("git_dirty", ckpt.get("provenance", {}).get("git_dirty")),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "manifest_path": str(dataset.manifest),
        "manifest_sha256": compute_file_sha256(dataset.manifest),
        "resolved_config_sha256": cfg_sha,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "mode": mode_tag,
        "tiling": args.tiling,
    }

    save_evaluation_artifacts(out_dir, rows, summary)

    # Save reliability_diagnostics.csv
    if diag_rows:
        with open(out_dir / "reliability_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)

    print(f"\nEvaluation Results:")
    print(f"  MAE: {summary['MAE']:.2f} | RMSE: {summary['RMSE']:.2f} | NAE: {summary['NAE']:.3f} | Bias: {summary['Bias']:+.2f}")
    if "mae_y0" in summary:
        print(f"  Iterates Full-Image MAE: Y0={summary['mae_y0']:.2f} -> Y1={summary['mae_y1']:.2f} -> Y2={summary['mae_y2']:.2f}")
    lo, hi = density_bins
    print(f"  Sparse MAE (<={lo:g}): {summary['mae_sparse']:.2f} (n={summary['n_sparse']})")
    print(f"  Moderate MAE ({lo:g}-{hi:g}): {summary['mae_moderate']:.2f} (n={summary['n_moderate']})")
    print(f"  Dense MAE (>{hi:g}): {summary['mae_dense']:.2f} (n={summary['n_dense']})")
    if "pearson_rate_var_error" in summary:
        print(f"  Pearson(var, err): {summary['pearson_rate_var_error']:.4f}")
        print(f"  Spearman(var, err): {summary['spearman_rate_var_error']:.4f}")
        print(f"  Spearman(weight, err): {summary['spearman_weight_error']:.4f}")
    if "coverage_95" in summary:
        print(f"  Coverage: 50%={summary['coverage_50']:.1%} (gap: {summary['calib_gap_50']:+.1%}), 80%={summary['coverage_80']:.1%} (gap: {summary['calib_gap_80']:+.1%}), 95%={summary['coverage_95']:.1%} (gap: {summary['calib_gap_95']:+.1%})")
    print(f"Saved artifacts to {out_dir}\n")


if __name__ == "__main__":
    main()
