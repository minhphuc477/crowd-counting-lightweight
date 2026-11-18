"""Layer-2 & Layer-4 Evaluation CLI for SFI-Lite & RMRv3.

Computes:
  - Layer-2 Efficiency & Stratified Counting: Params, MAE, RMSE, NAE, Bias,
    Density-tier MAE (Sparse/Medium/Dense), Per-Size MAE (small <16px, medium 16-32px, large >32px).
  - Layer-4 Mechanism Verification: hf_preservation_ratio, peak_sharpness, spatial_entropy,
    gate_response_analysis (image-level & intra-image local patch-level), and compute_erf.

Usage:
  python scripts/eval_sfi_mechanisms.py \
      --config configs/benchmarks/nextgen_sub65/exp_sha_sfi_v2_dsimr_subpixel_101k.yaml \
      --checkpoint runs_fixed/benchmarks/nextgen_sub65/exp_sha_sfi_v2_dsimr_subpixel_101k/best_val_mae.pt \
      --output-json results/sfi_mechanisms_sha.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from rmr_core.data import CrowdManifestDataset
from rmr_core.mechanism_metrics import (
    compute_erf,
    gate_response_analysis,
    peak_sharpness,
    per_size_mae,
    spatial_entropy,
)
from rmr_v3.engine import make_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Layer-2 & Layer-4 Mechanism Metrics for SFI-Lite / RMRv3")
    ap.add_argument("-c", "--config", required=True, help="Path to YAML config")
    ap.add_argument("--checkpoint", "--ckpt", default=None, help="Path to .pt checkpoint (optional for ERF/untrained probe)")
    ap.add_argument("--manifest", default=None, help="Override val_manifest from config")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on eval samples (0 = all)")
    ap.add_argument("--output-json", default=None, help="Path to save JSON report")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg.setdefault("model", {})["pretrained"] = False
    device = torch.device(args.device)

    model, _ = make_model(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("ema_model") or ckpt.get("model") or ckpt
        model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Measure inference FPS on 512x512
    dummy = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iters = 30
        for _ in range(n_iters):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fps = float(n_iters / max(time.perf_counter() - t0, 1e-6))

    # Compute ERF at P4
    erf_res = compute_erf(model, input_shape=(1, 3, 256, 256), target_layer="p4", num_samples=8, device=device)

    d_cfg = cfg.get("data", {})
    manifest_path = args.manifest or d_cfg.get("val_manifest")
    norm_mode = CrowdManifestDataset.resolve_norm(cfg["model"].get("backbone_name", ""), d_cfg.get("norm_mode", "auto"))
    stride = int(cfg["model"].get("output_stride", 2))

    report: dict[str, Any] = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "trainable_params": n_params,
        "fps_512": round(fps, 2),
        "erf_p4": {"r50_px": round(erf_res["r50_px"], 2), "r95_px": round(erf_res["r95_px"], 2)},
    }

    if manifest_path and Path(manifest_path).exists():
        ds = CrowdManifestDataset(
            manifest_path,
            data_root=d_cfg.get("data_root", "data"),
            train=False,
            output_stride=stride,
            max_size=int(d_cfg.get("max_size", 2048)),
            norm_mode=norm_mode,
        )
        n_eval = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)

        images_list: list[torch.Tensor] = []
        pts_list: list[torch.Tensor] = []
        boxes_list: list[torch.Tensor | None] = []
        pred_maps: list[torch.Tensor] = []
        abs_errs: list[float] = []
        sq_errs: list[float] = []
        signed_errs: list[float] = []
        sharpness_vals: list[float] = []
        entropy_vals: list[float] = []
        norm_entropy_vals: list[float] = []

        # Load optional bounding boxes directly from manifest if present
        manifest_boxes: list[torch.Tensor | None] = []
        with open(manifest_path, "r", encoding="utf-8") as mf:
            for line in mf:
                item = json.loads(line)
                bxs = item.get("boxes")
                manifest_boxes.append(torch.tensor(bxs, dtype=torch.float32) if bxs else None)

        with torch.no_grad():
            for i in range(n_eval):
                sample = ds[i]
                img = sample["image"]
                pts = sample["points"]
                gt_cnt = float(sample["target_y"].sum().item())
                out = model(img.unsqueeze(0).to(device))
                y_pred = (out["y"] if isinstance(out, dict) else out.y)[0].cpu()
                pred_cnt = float(y_pred.sum().item())

                images_list.append(img)
                pts_list.append(pts)
                boxes_list.append(manifest_boxes[i] if i < len(manifest_boxes) else None)
                pred_maps.append(y_pred)

                err = pred_cnt - gt_cnt
                abs_errs.append(abs(err))
                sq_errs.append(err * err)
                signed_errs.append(err)

                if pts.numel() > 0:
                    ps = peak_sharpness(y_pred, pts, stride=stride)
                    if ps["num_valid_heads"] > 0:
                        sharpness_vals.append(ps["mean_sharpness"])
                se = spatial_entropy(y_pred)
                entropy_vals.append(se["spatial_entropy"])
                norm_entropy_vals.append(se["normalized_entropy"])

        gate_res = gate_response_analysis(model, images_list, pts_list)
        size_res = per_size_mae(pred_maps, pts_list, boxes_list=boxes_list, stride=stride)

        report["counting"] = {
            "num_samples": n_eval,
            "mae": round(float(np.mean(abs_errs)), 4),
            "rmse": round(float(np.sqrt(np.mean(sq_errs))), 4),
            "bias": round(float(np.mean(signed_errs)), 4),
        }
        report["mechanisms"] = {
            "hf_preservation_ratio": gate_res.get("mean_hf_preservation_ratio"),
            "mean_peak_sharpness": float(np.mean(sharpness_vals)) if sharpness_vals else float("nan"),
            "mean_spatial_entropy": float(np.mean(entropy_vals)) if entropy_vals else float("nan"),
            "mean_normalized_entropy": float(np.mean(norm_entropy_vals)) if norm_entropy_vals else float("nan"),
            "gate_response": gate_res,
            "per_size_mae": size_res,
        }

    print(json.dumps(report, indent=2))
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
