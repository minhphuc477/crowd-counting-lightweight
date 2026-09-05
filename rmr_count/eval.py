from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import CrowdManifestDataset, collate_eval
from .metrics import bootstrap_ci, density_stratified_mae, game_single, summarize_predictions
from .model import RMRConfig, RMRCount
from .operators import regional_sum


def make_model_from_ckpt(ckpt: dict, device: torch.device) -> RMRCount:
    cfg = ckpt["config"]
    mcfg = RMRConfig(
        output_stride=cfg["model"].get("output_stride", 4),
        feature_width=cfg["model"].get("feature_width", 32),
        region_sizes_px=tuple(cfg["model"].get("region_sizes_px", [16, 32, 64, 128])),
        region_overlap=cfg["model"].get("region_overlap", 0.5),
        include_full_image=cfg["model"].get("include_full_image", True),
        iterations=cfg["model"].get("iterations", 2),
        eta_max=cfg["model"].get("eta_max", 0.20),
        eta_init=cfg["model"].get("eta_init", 0.05),
        residual_clip=cfg["model"].get("residual_clip", 5.0),
        # P0 fix: restore ablation flag — without this, RMR-Jacobian checkpoints are
        # silently evaluated as RMR-Latent (different update rule → wrong results).
        use_jacobian_gate=cfg["model"].get("use_jacobian_gate", False),
    )
    model = RMRCount(mcfg, variant=cfg["model"]["variant"])
    model.load_state_dict(ckpt["model"], strict=True)
    # P0 fix: restore solver_strength — without this, diagnostic checkpoints saved
    # during solver ramp-up (strength < 1.0) are loaded with strength=1.0, making
    # eval results not reproducible from the checkpoint's training epoch.
    # Production best_val_mae.pt checkpoints always have strength=1.0, so this is
    # a no-op for final results but is critical for mid-training diagnostics.
    saved_strength = float(ckpt.get("solver_strength", 1.0))
    model.set_solver_strength(saved_strength)
    return model.to(device).eval()


@torch.no_grad()
def predict_direct(
    model: RMRCount,
    image: torch.Tensor,
    target: torch.Tensor | None = None,
    region_mode: str = "predicted",
) -> tuple[torch.Tensor, dict]:
    """Direct prediction with optional mechanism diagnostics.

    region_mode:
      predicted : normal model
      oracle    : replace regional head output by exact GT regional counts (analysis only)
      shuffled  : shuffle predicted regional evidence within each scale family
    """
    x = image.unsqueeze(0)
    if region_mode == "predicted" or model.region_head is None:
        out = model(x)
    elif region_mode == "shuffled":
        out = model(x, shuffle_region=True)
    elif region_mode == "oracle":
        if target is None:
            raise ValueError("oracle region_mode requires target")
        # First pass is only to obtain the exact deterministic RegionSet.
        probe = model(x)
        regions = probe["regions"]
        b_gt = regional_sum(target.unsqueeze(0), regions.boxes)
        out = model(x, b_region_override=b_gt)
    else:
        raise ValueError(f"unknown region_mode={region_mode}")
    return out["y"][0], out


def _aligned_floor(v: int, stride: int) -> int:
    return (v // stride) * stride


def _aligned_ceil(v: int, stride: int) -> int:
    return ((v + stride - 1) // stride) * stride


@torch.no_grad()
def predict_tiled(
    model: RMRCount,
    image: torch.Tensor,
    tile_size: int = 512,
    halo: int = 0,
) -> torch.Tensor:
    """Core/halo tiled prediction assembled without double-counting.

    Core boundaries are aligned to output stride except the final image boundary.
    Halo affects context only; only the core prediction is written to the output.
    """
    _, h, w = image.shape
    s = model.cfg.output_stride
    tile_size = max(s, _aligned_floor(tile_size, s))
    halo = max(0, _aligned_floor(halo, s))
    gh, gw = math.ceil(h / s), math.ceil(w / s)
    canvas = image.new_zeros((1, gh, gw))

    ys = list(range(0, h, tile_size))
    xs = list(range(0, w, tile_size))
    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)

            sy0 = max(0, _aligned_floor(y0 - halo, s))
            sx0 = max(0, _aligned_floor(x0 - halo, s))
            sy1 = min(h, _aligned_ceil(y1 + halo, s))
            sx1 = min(w, _aligned_ceil(x1 + halo, s))
            patch = image[:, sy0:sy1, sx0:sx1].unsqueeze(0)
            y_patch = model(patch)["y"][0]

            gy0 = y0 // s
            gx0 = x0 // s
            gy1 = math.ceil(y1 / s)
            gx1 = math.ceil(x1 / s)
            ly0 = (y0 - sy0) // s
            lx0 = (x0 - sx0) // s
            hh = gy1 - gy0
            ww = gx1 - gx0
            canvas[:, gy0:gy1, gx0:gx1] = y_patch[:, ly0:ly0 + hh, lx0:lx0 + ww]
    return canvas


@torch.no_grad()
def evaluate(
    model: RMRCount,
    loader: DataLoader,
    device: torch.device,
    tile_size: int,
    practical_halo: int,
    region_mode: str = "predicted",
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    region_errors: dict[object, list[float]] = defaultdict(list)
    mechanism_errors: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    region_count_bins: dict[str, list[float]] = defaultdict(list)

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].to(device)
            target = sample["target_y"].to(device)
            y, out = predict_direct(model, image, target=target, region_mode=region_mode)
            if region_mode == "predicted":
                y_t0 = predict_tiled(model, image, tile_size=tile_size, halo=0)
                y_th = predict_tiled(model, image, tile_size=tile_size, halo=practical_halo)
            else:
                # Oracle/shuffled modes are mechanism diagnostics, not deployment metrics.
                y_t0 = y
                y_th = y

            gt = float(target.sum().item())
            pred = float(y.sum().item())
            pred_t0 = float(y_t0.sum().item())
            pred_th = float(y_th.sum().item())
            row = {
                "id": sample["id"],
                "gt": gt,
                "pred": pred,
                "pred_tiled_h0": pred_t0,
                "pred_tiled_practical": pred_th,
                "abs_err": abs(pred - gt),
                "direct_tiled_h0_abs": abs(pred - pred_t0),
                "direct_tiled_practical_abs": abs(pred - pred_th),
                "direct_tiled_h0_norm": abs(pred - pred_t0) / max(gt, 1.0),
                "direct_tiled_practical_norm": abs(pred - pred_th) / max(gt, 1.0),
            }
            for level in range(4):
                row[f"GAME{level}"] = game_single(y, target, level)
            rows.append(row)

            regions = out["regions"]
            p_reg = regional_sum(y.unsqueeze(0), regions.boxes)[0, 0]
            t_reg = regional_sum(target.unsqueeze(0), regions.boxes)[0, 0]
            ae = (p_reg - t_reg).abs()
            nmae = ae / t_reg.clamp_min(1.0)
            for sid in torch.unique(regions.scale_id):
                m = regions.scale_id == sid
                sid_i = int(sid.item())
                region_errors[sid_i].extend(ae[m].detach().cpu().tolist())
                region_errors[(sid_i, "nmae")].extend(
                    nmae[m].detach().cpu().tolist()
                )

            # Count-stratified regional diagnostics. These are more meaningful than
            # comparing a small-region MAE directly to whole-image MAE.
            gt_flat = t_reg.detach()
            ae_flat = ae.detach()
            bins = {
                "0": gt_flat == 0,
                "1": gt_flat == 1,
                "2_4": (gt_flat >= 2) & (gt_flat <= 4),
                "5_9": (gt_flat >= 5) & (gt_flat <= 9),
                "10p": gt_flat >= 10,
            }
            for name, mask in bins.items():
                if mask.any():
                    region_count_bins[name].extend(ae_flat[mask].cpu().tolist())

            # Mechanism trajectory: error to GT and disagreement with predicted b_R
            # at every iterate. This directly tests whether reconciliation reduces
            # regional inconsistency rather than merely changing the final count.
            iterates = out.get("iterates", [y])
            b_pred = out.get("b_region")
            for ti, yi in enumerate(iterates):
                q_i = regional_sum(yi.unsqueeze(0) if yi.ndim == 3 else yi, regions.boxes)[0, 0]
                gt_ae_i = (q_i - t_reg).abs()
                for sid in torch.unique(regions.scale_id):
                    m = regions.scale_id == sid
                    sid_i = int(sid.item())
                    mechanism_errors[(ti, sid_i, "gt_mae")].extend(
                        gt_ae_i[m].detach().cpu().tolist()
                    )
                if b_pred is not None:
                    b_i = b_pred[0, 0]
                    pred_dis_i = (q_i - b_i).abs()
                    for sid in torch.unique(regions.scale_id):
                        m = regions.scale_id == sid
                        sid_i = int(sid.item())
                        mechanism_errors[(ti, sid_i, "pred_disagreement")].extend(
                            pred_dis_i[m].detach().cpu().tolist()
                        )

    summary = summarize_predictions(rows)
    summary.update(density_stratified_mae(rows))
    summary["DirectTiledH0_MeanAbs"] = float(np.mean([r["direct_tiled_h0_abs"] for r in rows]))
    summary["DirectTiledH0_MeanNorm"] = float(np.mean([r["direct_tiled_h0_norm"] for r in rows]))
    summary["DirectTiledPractical_MeanAbs"] = float(np.mean([r["direct_tiled_practical_abs"] for r in rows]))
    summary["DirectTiledPractical_MeanNorm"] = float(np.mean([r["direct_tiled_practical_norm"] for r in rows]))

    paired = np.asarray([r["direct_tiled_practical_norm"] for r in rows], dtype=np.float64)
    lo, hi = bootstrap_ci(paired, n_boot=5000)
    summary["DirectTiledPractical_MeanNorm_CI95_lo"] = lo
    summary["DirectTiledPractical_MeanNorm_CI95_hi"] = hi

    for sid_key, vals in region_errors.items():
        if isinstance(sid_key, tuple):
            sid, kind = sid_key
            name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
            summary[f"RegionNMAE_px_{name}"] = float(np.mean(vals)) if vals else float("nan")
        else:
            sid = sid_key
            name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
            summary[f"RegionMAE_px_{name}"] = float(np.mean(vals)) if vals else float("nan")
    for name, vals in region_count_bins.items():
        summary[f"RegionMAE_countbin_{name}"] = float(np.mean(vals)) if vals else float("nan")

    for (ti, sid, kind), vals in mechanism_errors.items():
        scale_name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
        summary[f"Iter{ti}_{kind}_px_{scale_name}"] = (
            float(np.mean(vals)) if vals else float("nan")
        )

    summary["region_mode"] = region_mode
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--practical-halo", type=int, default=64)
    ap.add_argument("--region-mode", choices=["predicted", "oracle", "shuffled"], default="predicted")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = make_model_from_ckpt(ckpt, device)
    ds = CrowdManifestDataset(args.manifest, train=False, output_stride=model.cfg.output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

    rows, summary = evaluate(
        model, loader, device, args.tile_size, args.practical_halo, region_mode=args.region_mode
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
