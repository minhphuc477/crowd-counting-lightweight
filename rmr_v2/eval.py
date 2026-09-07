from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset, save_evaluation_artifacts
from .model import RMRConfig, RMRCount


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

    manifest_path = Path(args.manifest)
    dataset = CrowdManifestDataset(manifest_path, train=False, output_stride=model.cfg.output_stride)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / f"eval_{manifest_path.stem}"

    print(f"Evaluating {ckpt_path.name} on {manifest_path} ({len(dataset)} samples)...", flush=True)

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=model.cfg.output_stride,
        run_tiling=not args.no_tiling,
        tile_size=args.tile_size,
        practical_halo=args.halo,
    )

    save_evaluation_artifacts(out_dir, rows, summary)
    print(f"Evaluation complete. MAE: {summary['MAE']:.2f}, RMSE: {summary['RMSE']:.2f}, NAE: {summary['NAE']:.4f}")
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
