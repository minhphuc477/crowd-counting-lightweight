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


def make_model_from_ckpt(
    ckpt: dict,
    device: torch.device,
) -> RMRCount:
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    # --------------------------------------------------------------
    # Backward compatibility.
    # Historical RMR checkpoints did not store update_rule.
    # They are RMR-Latent unless use_jacobian_gate=True.
    # --------------------------------------------------------------
    update_rule = model_cfg.get("update_rule")
    if update_rule is None:
        update_rule = (
            "jacobian"
            if model_cfg.get("use_jacobian_gate", False)
            else "latent"
        )

    mcfg = RMRConfig(
        output_stride=model_cfg.get(
            "output_stride",
            4,
        ),
        feature_width=model_cfg.get(
            "feature_width",
            32,
        ),

        # Use current registered geometry as fallback.
        region_sizes_px=tuple(
            model_cfg.get(
                "region_sizes_px",
                [32, 64, 128],
            )
        ),
        region_overlap=model_cfg.get(
            "region_overlap",
            0.5,
        ),
        include_full_image=model_cfg.get(
            "include_full_image",
            False,
        ),

        iterations=model_cfg.get(
            "iterations",
            2,
        ),

        eta_max=model_cfg.get(
            "eta_max",
            0.20,
        ),
        eta_init=model_cfg.get(
            "eta_init",
            0.05,
        ),
        residual_clip=model_cfg.get(
            "residual_clip",
            5.0,
        ),

        update_rule=update_rule,
        use_jacobian_gate=model_cfg.get(
            "use_jacobian_gate",
            False,
        ),

        sirt_omega=model_cfg.get(
            "sirt_omega",
            1.0,
        ),
        learnable_sirt_omega=model_cfg.get(
            "learnable_sirt_omega",
            False,
        ),
        projected_use_preconditioner=model_cfg.get(
            "projected_use_preconditioner",
            False,
        ),
        detach_region_evidence=model_cfg.get(
            "detach_region_evidence",
            True,
        ),
    )

    model = RMRCount(
        mcfg,
        variant=model_cfg["variant"],
    )

    state_dict = dict(ckpt["model"])
    if getattr(model, "eta_logits", None) is None and "eta_logits" in state_dict:
        state_dict.pop("eta_logits", None)
    if getattr(model, "log_sirt_omega", None) is None and "log_sirt_omega" in state_dict:
        state_dict.pop("log_sirt_omega", None)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    saved_strength = float(
        ckpt.get(
            "solver_strength",
            1.0,
        )
    )
    model.set_solver_strength(
        saved_strength
    )

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
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    rows: list[dict] = []
    region_errors: dict[object, list[float]] = defaultdict(list)
    mechanism_errors: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    region_count_bins: dict[str, list[float]] = defaultdict(list)
    regional_trace_rows: list[dict] = []
    solver_trace_rows: list[dict] = []

    # Extra diagnostic accumulators for b_R vs GT
    b_head_errors: dict[object, list[float]] = defaultdict(list)
    all_b_preds: list[float] = []
    all_gt_regions: list[float] = []
    zero_region_mass_sum: float = 0.0
    zero_region_count: int = 0
    pos_region_miss_count: int = 0
    pos_region_total: int = 0

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
            # Physical-support GAME when raw points and image dimensions are provided
            if "points" in sample and "height" in sample and "width" in sample:
                from .metrics import game_physical_image
                game_dict = game_physical_image(
                    y,
                    sample["points"],
                    image_h=sample["height"],
                    image_w=sample["width"],
                    stride=model.cfg.output_stride,
                    levels=(0, 1, 2, 3),
                )
                for level in range(4):
                    row[f"GAME{level}"] = game_dict[level]
            else:
                for level in range(4):
                    row[f"GAME{level}"] = game_single(y, target, level)
            rows.append(row)

            # P0 regression fix: Baseline controls (direct, local_refine) do not construct
            # regions during forward. The evaluator constructs diagnostic RegionSet after
            # inference timing so baseline latency is never inflated.
            regions = out.get("regions")
            if regions is None:
                regions = model._regions(y.shape[-2], y.shape[-1], y.device)

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

            # Count-stratified regional diagnostics.
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

            # Background hallucination & positive miss diagnostics
            zero_mask = gt_flat == 0
            if zero_mask.any():
                zero_region_mass_sum += float(p_reg[zero_mask].sum().item())
                zero_region_count += int(zero_mask.sum().item())
            pos_mask = gt_flat > 0
            if pos_mask.any():
                pos_region_total += int(pos_mask.sum().item())
                pos_region_miss_count += int((p_reg[pos_mask] < 0.1).sum().item())

            # Mechanism trajectory: error to GT, disagreement with b_R, and solver energy
            iterates = out.get("iterates", [y])
            residuals = out.get("residual_fields", [])
            preconditioners = out.get("preconditioner_fields", [])
            step_sizes = out.get("step_sizes", [])
            b_pred = out.get("b_region")

            if b_pred is not None:
                b_i = b_pred[0, 0]
                ae_b = (b_i - t_reg).abs()
                all_b_preds.extend(b_i.detach().cpu().tolist())
                all_gt_regions.extend(t_reg.detach().cpu().tolist())
                for sid in torch.unique(regions.scale_id):
                    m = regions.scale_id == sid
                    sid_i = int(sid.item())
                    b_head_errors[sid_i].extend(ae_b[m].detach().cpu().tolist())

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

            # Solver energy trajectory: E_a(Y) = 0.5 * sum_R ((q_R - b_R)^2 / |R|)
            def _energy(curr_y: torch.Tensor, b_t: torch.Tensor) -> float:
                q = regional_sum(curr_y.unsqueeze(0) if curr_y.ndim == 3 else curr_y, regions.boxes)[0, 0]
                diff = q - b_t[0, 0]
                area_m = regions.area.clamp_min(1.0).to(diff.device)
                return float((0.5 * (diff ** 2) / area_m).sum().item())

            for t in range(len(iterates) - 1):
                y_curr = iterates[t]
                y_next = iterates[t + 1]
                eta_val = step_sizes[t] if t < len(step_sizes) else 0.0
                r_val = residuals[t] if t < len(residuals) else None
                m_val = preconditioners[t] if t < len(preconditioners) else None

                e_before = _energy(y_curr, b_pred) if b_pred is not None else float("nan")
                e_after = _energy(y_next, b_pred) if b_pred is not None else float("nan")

                r_abs = r_val.abs() if r_val is not None else None
                r_mean = float(r_abs.mean().item()) if r_abs is not None else 0.0
                r_max = float(r_abs.max().item()) if r_abs is not None else 0.0
                clip_frac = float((r_abs >= (model.cfg.residual_clip - 1e-4)).float().mean().item()) if r_abs is not None and model.cfg.residual_clip > 0 else 0.0

                m_mean = float(m_val.mean().item()) if m_val is not None else float("nan")
                m_std = float(m_val.std().item()) if m_val is not None and m_val.numel() > 1 else float("nan")
                m_min = float(m_val.min().item()) if m_val is not None else float("nan")
                m_max = float(m_val.max().item()) if m_val is not None else float("nan")

                d_count = float(y_next.sum().item() - y_curr.sum().item())
                d_l1 = float((y_next - y_curr).abs().sum().item())

                solver_trace_rows.append({
                    "image_id": sample["id"],
                    "iteration": t,
                    "eta": eta_val,
                    "energy_before": e_before,
                    "energy_after": e_after,
                    "energy_decreased": int(e_after < e_before) if not math.isnan(e_before) else 0,
                    "residual_mean": r_mean,
                    "residual_max": r_max,
                    "clip_fraction": clip_frac,
                    "M_mean": m_mean,
                    "M_std": m_std,
                    "M_min": m_min,
                    "M_max": m_max,
                    "delta_count": d_count,
                    "delta_measure_l1": d_l1,
                })

            # Long-table regional trace
            if b_pred is not None:
                b_val_flat = b_pred[0, 0].detach().cpu().numpy()
                q_val_flat = p_reg.detach().cpu().numpy()
                gt_val_flat = t_reg.detach().cpu().numpy()
                area_flat = regions.area.detach().cpu().numpy()
                scale_id_flat = regions.scale_id.detach().cpu().numpy()

                for reg_idx in range(len(b_val_flat)):
                    sid = int(scale_id_flat[reg_idx])
                    scale_px = "full" if sid == -1 else model.cfg.region_sizes_px[sid]
                    c_res = float(q_val_flat[reg_idx] - b_val_flat[reg_idx])
                    r_res = float(c_res / max(area_flat[reg_idx], 1.0))
                    regional_trace_rows.append({
                        "image_id": sample["id"],
                        "scale_px": scale_px,
                        "region_id": reg_idx,
                        "gt_count": float(gt_val_flat[reg_idx]),
                        "b_pred": float(b_val_flat[reg_idx]),
                        "q_pred": float(q_val_flat[reg_idx]),
                        "count_residual": c_res,
                        "rate_residual": r_res,
                    })

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

    # Regional head b_R accuracy metrics
    for sid_i, vals in b_head_errors.items():
        scale_name = "full" if sid_i == -1 else str(model.cfg.region_sizes_px[sid_i])
        summary[f"BHead_MAE_px_{scale_name}"] = float(np.mean(vals)) if vals else float("nan")
    if all_b_preds and all_gt_regions:
        arr_b = np.asarray(all_b_preds, dtype=np.float64)
        arr_gt = np.asarray(all_gt_regions, dtype=np.float64)
        summary["BHead_Overall_MAE"] = float(np.mean(np.abs(arr_b - arr_gt)))
        if arr_b.std() > 1e-6 and arr_gt.std() > 1e-6:
            r_corr = np.corrcoef(arr_b, arr_gt)[0, 1]
            summary["BHead_Pearson_Correlation"] = float(r_corr)
        else:
            summary["BHead_Pearson_Correlation"] = float("nan")

    # False mass & positive miss
    summary["ZeroRegion_FalseMass_PerRegion"] = (
        zero_region_mass_sum / max(1, zero_region_count)
    )
    summary["PositiveRegion_MissRate"] = (
        pos_region_miss_count / max(1, pos_region_total)
    )

    # Solver energy reduction rate
    if solver_trace_rows:
        decreased = [s["energy_decreased"] for s in solver_trace_rows if not math.isnan(s["energy_before"])]
        summary["Solver_Energy_Reduction_Rate"] = float(np.mean(decreased)) if decreased else float("nan")

    summary["region_mode"] = region_mode
    return rows, summary, regional_trace_rows, solver_trace_rows


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

    rows, summary, regional_trace, solver_trace = evaluate(
        model, loader, device, args.tile_size, args.practical_halo, region_mode=args.region_mode
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if regional_trace:
        with (out_dir / "regional_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(regional_trace[0].keys()))
            writer.writeheader()
            writer.writerows(regional_trace)

    if solver_trace:
        with (out_dir / "solver_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(solver_trace[0].keys()))
            writer.writeheader()
            writer.writerows(solver_trace)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
