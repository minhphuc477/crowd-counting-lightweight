"""JRTIP Qualitative Visual Comparison Figure Generator (300 DPI).

Compares Input + GT Points | Baseline (e.g. V0 Additive) | Proposed LFI-99k
across multi-regime scenes (Empty N=0, Sparse/Moderate, Dense, and Failure Case).

Usage:
  python scripts/render_jrtip_qualitative.py ^
      --lfi-ckpt runs_fixed/benchmarks/lfi_sub100k/exp_sha_lfi_dsimr_99k/best_val_mae.pt ^
      --baseline-ckpt runs_fixed/experiments_pr_sfi/axis_a_v0_additive/best_val_mae.pt ^
      --manifest data/sha_a_test.jsonl ^
      --output-path results/figures/jrtip_qualitative_sha.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from rmr_core.data import CrowdManifestDataset
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def select_representative_samples(manifest_path: str, explicit_ids: list[str] | None) -> list[tuple[str, dict]]:
    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if explicit_ids:
        by_id = {str(s.get("id", Path(s["image"]).stem)): s for s in samples}
        picked = []
        for sid in explicit_ids:
            if sid in by_id:
                n = len(by_id[sid].get("points", []))
                picked.append((f"ID={sid} (GT={n})", by_id[sid]))
        if picked:
            return picked

    samples_sorted = sorted(samples, key=lambda s: len(s.get("points", [])))
    empty = [s for s in samples_sorted if len(s.get("points", [])) == 0]
    sparse = [s for s in samples_sorted if 15 <= len(s.get("points", [])) <= 100]
    mod = [s for s in samples_sorted if 150 <= len(s.get("points", [])) <= 450]
    dense = [s for s in samples_sorted if len(s.get("points", [])) >= 600]

    out: list[tuple[str, dict]] = []
    if empty:
        out.append(("Negative (N=0)", empty[0]))
    if sparse:
        out.append(("Sparse", sparse[len(sparse) // 2]))
    if mod:
        out.append(("Moderate", mod[len(mod) // 2]))
    if dense:
        out.append(("Dense", dense[len(dense) // 2]))
    out.append(("Extreme / Outlier", samples_sorted[-1]))
    return out[:4]


@torch.no_grad()
def predict_density_map(
    model: torch.nn.Module,
    raw_img: Image.Image,
    norm_mode: str,
    device: torch.device,
    max_dim_cap: int = 1536,
) -> tuple[np.ndarray, np.ndarray, float]:
    w_orig, h_orig = raw_img.size
    scale = min(1.0, float(max_dim_cap) / float(max(w_orig, h_orig)))
    w_disp = int(round(w_orig * scale))
    h_disp = int(round(h_orig * scale))
    disp_img = raw_img.resize((w_disp, h_disp), Image.Resampling.BILINEAR) if scale < 1.0 else raw_img

    pad_h = (32 - h_disp % 32) % 32
    pad_w = (32 - w_disp % 32) % 32
    arr = np.asarray(disp_img, dtype=np.float32) / 255.0
    t_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    t_img = normalize_image(t_img, mode=norm_mode)
    if pad_h > 0 or pad_w > 0:
        t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    out = model(t_img)
    y = out["y"] if "y" in out else out["y0"]
    y_crop = y[0, 0, : max(h_disp // 2, 1), : max(w_disp // 2, 1)].float().cpu().numpy()
    return np.asarray(disp_img), y_crop, float(scale)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 300-DPI JRTIP qualitative density comparison panels")
    ap.add_argument("--lfi-ckpt", required=True, help="Path to LFI-99k best_val_mae.pt")
    ap.add_argument("--baseline-ckpt", default=None, help="Optional baseline checkpoint (e.g. V0 Additive)")
    ap.add_argument("--manifest", required=True, help="Path to evaluation manifest jsonl")
    ap.add_argument("--ids", nargs="+", default=None, help="Optional explicit sample IDs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-path", default="results/figures/jrtip_qualitative.png")
    args = ap.parse_args()

    device = torch.device(args.device)
    lfi_model, _, lfi_cfg, _ = load_model_from_ckpt(Path(args.lfi_ckpt), device=device)
    lfi_model.eval()
    lfi_norm = CrowdManifestDataset.resolve_norm(
        lfi_cfg.get("model", {}).get("backbone_name", ""),
        lfi_cfg.get("data", {}).get("norm_mode", "auto"),
    )

    base_model, base_norm = None, lfi_norm
    if args.baseline_ckpt:
        base_model, _, base_cfg, _ = load_model_from_ckpt(Path(args.baseline_ckpt), device=device)
        base_model.eval()
        base_norm = CrowdManifestDataset.resolve_norm(
            base_cfg.get("model", {}).get("backbone_name", ""),
            base_cfg.get("data", {}).get("norm_mode", "auto"),
        )

    selected = select_representative_samples(args.manifest, args.ids)
    n_rows = len(selected)
    n_cols = 4 if base_model is not None else 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), dpi=300)
    if n_rows == 1:
        axes = axes[None, :]
    plt.subplots_adjust(wspace=0.06, hspace=0.22)

    for r_idx, (tier_label, sample) in enumerate(selected):
        img_rel = sample["image"]
        img_path = Path(args.data_root) / img_rel if not Path(img_rel).exists() else Path(img_rel)
        raw_img = Image.open(img_path).convert("RGB")
        pts = np.asarray(sample.get("points", []), dtype=np.float32)
        gt_cnt = int(len(pts))

        disp_arr, y_lfi, scale = predict_density_map(lfi_model, raw_img, lfi_norm, device)
        pred_lfi = float(y_lfi.sum())

        # Col 0: Input RGB
        axes[r_idx, 0].imshow(disp_arr)
        axes[r_idx, 0].set_title(f"{tier_label}\nInput (GT={gt_cnt})", fontsize=9.5, fontweight="bold")
        axes[r_idx, 0].axis("off")

        # Col 1: GT Points
        axes[r_idx, 1].imshow(disp_arr, alpha=0.75)
        if gt_cnt > 0:
            pts_s = pts * scale
            axes[r_idx, 1].scatter(pts_s[:, 0], pts_s[:, 1], s=4, c="#00FF66", edgecolors="none")
        axes[r_idx, 1].set_title(f"GT Points (N={gt_cnt})", fontsize=9.5, fontweight="bold")
        axes[r_idx, 1].axis("off")

        if base_model is not None:
            _, y_base, _ = predict_density_map(base_model, raw_img, base_norm, device)
            pred_base = float(y_base.sum())
            vmax = float(max(np.percentile(y_lfi, 99.7), np.percentile(y_base, 99.7), 1e-4))
            axes[r_idx, 2].imshow(y_base, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[r_idx, 2].set_title(f"V0 Additive: {pred_base:.1f}\n(Err={pred_base - gt_cnt:+.1f})", fontsize=9.5)
            axes[r_idx, 2].axis("off")

            axes[r_idx, 3].imshow(y_lfi, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[r_idx, 3].set_title(f"LFI-99k (Ours): {pred_lfi:.1f}\n(Err={pred_lfi - gt_cnt:+.1f})", fontsize=9.5, fontweight="bold")
            axes[r_idx, 3].axis("off")
        else:
            vmax = float(max(np.percentile(y_lfi, 99.7), 1e-4))
            axes[r_idx, 2].imshow(y_lfi, cmap="turbo", vmin=0.0, vmax=vmax)
            axes[r_idx, 2].set_title(f"LFI-99k (Ours): {pred_lfi:.1f}\n(Err={pred_lfi - gt_cnt:+.1f})", fontsize=9.5, fontweight="bold")
            axes[r_idx, 2].axis("off")

    out_png = Path(args.output_path)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved JRTIP qualitative figure: {out_png} and {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
