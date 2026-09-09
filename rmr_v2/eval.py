from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset, save_evaluation_artifacts
from .model import RMRConfig, RMRCount


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


def make_model_from_ckpt(ckpt: dict, device: torch.device) -> RMRCount:
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    update_rule = model_cfg.get("update_rule")
    if update_rule is None:
        update_rule = "jacobian" if model_cfg.get("use_jacobian_gate", False) else "latent"

    mcfg = RMRConfig(
        output_stride=model_cfg.get("output_stride", 4),
        feature_width=model_cfg.get("feature_width", 32),
        region_sizes_px=tuple(model_cfg.get("region_sizes_px", [32, 64, 128])),
        region_overlap=model_cfg.get("region_overlap", 0.5),
        include_full_image=model_cfg.get("include_full_image", False),
        iterations=model_cfg.get("iterations", 2),
        eta_max=model_cfg.get("eta_max", 0.20),
        eta_init=model_cfg.get("eta_init", 0.05),
        residual_clip=model_cfg.get("residual_clip", 5.0),
        update_rule=update_rule,
        use_jacobian_gate=model_cfg.get("use_jacobian_gate", False),
        sirt_omega=model_cfg.get("sirt_omega", 1.0),
        learnable_sirt_omega=model_cfg.get("learnable_sirt_omega", False),
        projected_use_preconditioner=model_cfg.get("projected_use_preconditioner", False),
        detach_region_evidence=model_cfg.get("detach_region_evidence", True),
        backbone_name=model_cfg.get(
            "backbone_name",
            model_cfg.get(
                "backbone",
                "tiny" if any(k.startswith("encoder.stem") for k in ckpt.get("model", {})) else "mobilenetv4_conv_small_050.e3000_r224_in1k",
            ),
        ),
        pretrained=False,
        init_m0=float(model_cfg.get("init_m0", 0.015763)),
        backbone_lr_scale=model_cfg.get("backbone_lr_scale", 0.1),
    )

    model = RMRCount(mcfg, variant=model_cfg["variant"])
    state_dict = dict(ckpt["model"])
    if getattr(model, "eta_logits", None) is None and "eta_logits" in state_dict:
        state_dict.pop("eta_logits", None)
    if getattr(model, "log_sirt_omega", None) is None and "log_sirt_omega" in state_dict:
        state_dict.pop("log_sirt_omega", None)

    model.load_state_dict(state_dict, strict=True)
    saved_strength = float(ckpt.get("solver_strength", 1.0))
    model.set_solver_strength(saved_strength)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate RMR-v2 checkpoint")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--manifest", default="data/sha_a_test.jsonl", help="Evaluation manifest jsonl")
    ap.add_argument("--output-dir", default=None, help="Directory to save artifacts")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--halo", type=int, default=64)
    ap.add_argument("--no-tiling", action="store_true", help="Disable tiled prediction")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model = make_model_from_ckpt(ckpt, device)
    cfg = ckpt.get("config", {})

    manifest_path = Path(args.manifest)
    dataset = CrowdManifestDataset(
        manifest_path,
        train=False,
        output_stride=model.cfg.output_stride,
        data_root=cfg.get("data", {}).get("data_root"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / f"eval_{manifest_path.stem}"

    print(f"Evaluating {ckpt_path.name} on {manifest_path} ({len(dataset)} samples)...", flush=True)

    density_bins = tuple(float(x) for x in cfg.get("eval", {}).get("density_bins", [100.0, 500.0]))

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=model.cfg.output_stride,
        run_tiling=not args.no_tiling,
        tile_size=args.tile_size,
        practical_halo=args.halo,
        enforce_gt_consistency=True,
        density_bins=density_bins,
    )

    eval_commit, eval_dirty = get_git_info()
    resolved_cfg_file = ckpt_path.parent / "resolved_config.yaml"
    if resolved_cfg_file.exists():
        cfg_sha = compute_file_sha256(resolved_cfg_file)
    else:
        cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()

    model_cfg = cfg.get("model", {})
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
        "variant": model_cfg.get("variant", "b5_projected"),
        "tiling": not args.no_tiling,
    }

    save_evaluation_artifacts(out_dir, rows, summary)
    lo, hi = density_bins
    print(f"\nEvaluation complete:")
    print(f"  MAE: {summary['MAE']:.2f} | RMSE: {summary['RMSE']:.2f} | NAE: {summary['NAE']:.4f} | Bias: {summary['Bias']:+.2f}")
    print(f"  GAME0: {summary['GAME0']:.2f} | GAME1: {summary['GAME1']:.2f} | GAME2: {summary['GAME2']:.2f} | GAME3: {summary['GAME3']:.2f}")
    print(f"  Sparse MAE (<={lo:g}): {summary.get('mae_sparse', 0.0):.2f} (n={summary.get('n_sparse', 0)})")
    print(f"  Moderate MAE ({lo:g}-{hi:g}): {summary.get('mae_moderate', 0.0):.2f} (n={summary.get('n_moderate', 0)})")
    print(f"  Dense MAE (>{hi:g}): {summary.get('mae_dense', 0.0):.2f} (n={summary.get('n_dense', 0)})")
    print(f"Artifacts saved to {out_dir}\n")


if __name__ == "__main__":
    main()
