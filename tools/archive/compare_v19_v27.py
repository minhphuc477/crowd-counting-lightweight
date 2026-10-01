#!/usr/bin/env python3
"""Per-image and regime comparison tool: RMR-v19 Benchmark vs RMR-v27 Candidates.

Analyzes test set predictions on ShanghaiTech Part A (182 images):
- Overall MAE, RMSE, WAPE, Pearson r
- Density regimes: Sparse (<=100), Moderate (101-500), Dense (>500)
- Paired statistical comparison (Wilcoxon signed-rank / paired t-test)
- Top regressions and top improvements
- Markdown report generation
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def find_predictions_csv(base_dir: Path, prefer_tta: bool = True) -> Path | None:
    """Find the best predictions.csv in an evaluation folder."""
    if not base_dir.exists():
        return None
    if base_dir.is_file() and base_dir.name == "predictions.csv":
        return base_dir

    subdirs = sorted(base_dir.glob("eval_*"), reverse=True)
    if prefer_tta:
        tta_dirs = [d for d in subdirs if "tta" in d.name.lower()]
        for d in tta_dirs:
            p = d / "predictions.csv"
            if p.exists():
                return p

    for d in subdirs:
        p = d / "predictions.csv"
        if p.exists():
            return p

    p = base_dir / "predictions.csv"
    if p.exists():
        return p
    return None


def load_preds(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load predictions keyed by image identifier."""
    preds: dict[str, dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_id = r.get("img_name") or r.get("image_name") or r.get("img_id") or r.get("id") or ""
            if not img_id and "image" in r:
                img_id = Path(r["image"]).name
            if not img_id:
                continue

            gt = float(r.get("gt_count", r.get("gt", 0.0)))
            pred = float(r.get("pred_count", r.get("pred", 0.0)))
            ae = abs(pred - gt)
            se = (pred - gt) ** 2
            preds[img_id] = {"gt": gt, "pred": pred, "ae": ae, "se": se}
    return preds


def compute_metrics(records: list[dict[str, Any]], key_prefix: str) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE, and Pearson r."""
    if not records:
        return {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "r": 0.0}
    n = len(records)
    sum_gt = sum(r["gt"] for r in records)
    sum_ae = sum(r[f"{key_prefix}_ae"] for r in records)
    sum_se = sum(r[f"{key_prefix}_se"] for r in records)

    mae = sum_ae / n
    rmse = math.sqrt(sum_se / n)
    wape = (sum_ae / max(sum_gt, 1e-6)) * 100.0

    mean_gt = sum_gt / n
    mean_pred = sum(r[f"{key_prefix}_pred"] for r in records) / n
    cov = sum((r["gt"] - mean_gt) * (r[f"{key_prefix}_pred"] - mean_pred) for r in records)
    std_gt = math.sqrt(sum((r["gt"] - mean_gt) ** 2 for r in records) + 1e-8)
    std_pred = math.sqrt(sum((r[f"{key_prefix}_pred"] - mean_pred) ** 2 for r in records) + 1e-8)
    r_val = cov / (std_gt * std_pred) if (std_gt * std_pred) > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "wape": wape, "r": r_val}


def paired_stats(diffs: list[dict[str, Any]]) -> dict[str, float]:
    """Compute paired mean error delta and standard error."""
    deltas = [d["v27_ae"] - d["v19_ae"] for d in diffs]
    n = len(deltas)
    if n == 0:
        return {"mean_delta": 0.0, "std_delta": 0.0, "t_stat": 0.0}
    mean_d = sum(deltas) / n
    var_d = sum((x - mean_d) ** 2 for x in deltas) / max(1, n - 1)
    std_d = math.sqrt(var_d)
    se_d = std_d / math.sqrt(n)
    t_stat = mean_d / se_d if se_d > 0 else 0.0
    return {"mean_delta": mean_d, "std_delta": std_d, "se_delta": se_d, "t_stat": t_stat}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RMR-v19 with RMR-v27 models")
    parser.add_argument(
        "--v19-path",
        type=str,
        default="runs/sha_a/rmr_v19_canonical_isotropic/eval_sha_a_test_weighted_tta/predictions.csv",
        help="Path to v19 predictions CSV or run directory",
    )
    parser.add_argument(
        "--v27-path",
        type=str,
        default="runs/sha_a/rmr_v27_canonical_restored",
        help="Path to v27 predictions CSV or run directory",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Number of extreme images to display")
    parser.add_argument("--output-md", type=str, default="", help="Optional markdown report path")
    args = parser.parse_args()

    v19_p = Path(args.v19_path)
    if v19_p.is_dir():
        v19_p = find_predictions_csv(v19_p, prefer_tta=True) or v19_p

    v27_p = Path(args.v27_path)
    if v27_p.is_dir():
        v27_p = find_predictions_csv(v27_p, prefer_tta=True) or v27_p

    if not v19_p or not v19_p.exists():
        print(f"[ERROR] v19 predictions not found: {v19_p}")
        return
    if not v27_p or not v27_p.exists():
        print(f"[WARNING] v27 predictions not found yet: {v27_p}")
        print("Run training and evaluation first before calling compare_v19_v27.")
        return

    v19 = load_preds(v19_p)
    v27 = load_preds(v27_p)

    common_keys = sorted(set(v19.keys()) & set(v27.keys()))
    if not common_keys:
        print(f"[ERROR] No overlapping images between v19 ({len(v19)}) and v27 ({len(v27)})")
        return

    diffs = []
    for k in common_keys:
        gt = v19[k]["gt"]
        p19 = v19[k]["pred"]
        ae19 = v19[k]["ae"]
        se19 = v19[k]["se"]

        p27 = v27[k]["pred"]
        ae27 = v27[k]["ae"]
        se27 = v27[k]["se"]

        diff = ae27 - ae19  # positive: v27 worse, negative: v27 better
        diffs.append({
            "id": k,
            "gt": gt,
            "v19_pred": p19,
            "v19_ae": ae19,
            "v19_se": se19,
            "v27_pred": p27,
            "v27_ae": ae27,
            "v27_se": se27,
            "delta_ae": diff,
        })

    diffs.sort(key=lambda x: x["delta_ae"], reverse=True)

    sparse = [d for d in diffs if d["gt"] <= 100]
    mod = [d for d in diffs if 100 < d["gt"] <= 500]
    dense = [d for d in diffs if d["gt"] > 500]

    stats = paired_stats(diffs)
    m19_all = compute_metrics(diffs, "v19")
    m27_all = compute_metrics(diffs, "v27")

    print("\n" + "=" * 100)
    print(f"RMR-v19 vs RMR-v27 BENCHMARK COMPARISON ({len(diffs)} images)")
    print(f"v19: {v19_p}")
    print(f"v27: {v27_p}")
    print("=" * 100)
    print(f"{'Metric':<15} | {'RMR-v19':<12} | {'RMR-v27':<12} | {'Delta':<12} | {'Status':<15}")
    print("-" * 100)
    mae_diff = m27_all["mae"] - m19_all["mae"]
    rmse_diff = m27_all["rmse"] - m19_all["rmse"]
    wape_diff = m27_all["wape"] - m19_all["wape"]
    r_diff = m27_all["r"] - m19_all["r"]

    status_mae = "BEATS v19" if mae_diff < 0 else "TRAILS v19"
    status_rmse = "BEATS v19" if rmse_diff < 0 else "TRAILS v19"

    print(f"{'MAE':<15} | {m19_all['mae']:<12.2f} | {m27_all['mae']:<12.2f} | {mae_diff:<+12.2f} | {status_mae:<15}")
    print(f"{'RMSE':<15} | {m19_all['rmse']:<12.2f} | {m27_all['rmse']:<12.2f} | {rmse_diff:<+12.2f} | {status_rmse:<15}")
    print(f"{'WAPE (%)':<15} | {m19_all['wape']:<12.2f} | {m27_all['wape']:<12.2f} | {wape_diff:<+12.2f} |")
    print(f"{'Pearson r':<15} | {m19_all['r']:<12.4f} | {m27_all['r']:<12.4f} | {r_diff:<+12.4f} |")
    print("-" * 100)
    print(f"Paired Mean Delta: {stats['mean_delta']:+.2f} ± {stats['se_delta']:.2f} (t-stat: {stats['t_stat']:+.2f})")

    print("\n" + "=" * 100)
    print("DENSITY REGIME BREAKDOWN: MAE")
    print("=" * 100)
    print(f"{'Regime':<22} | {'Count':<6} | {'v19 MAE':<10} | {'v27 MAE':<10} | {'Delta':<10} | {'Status'}")
    print("-" * 100)
    regimes = [
        ("Sparse (<=100)", sparse),
        ("Moderate (101-500)", mod),
        ("Dense (>500)", dense),
        ("Overall", diffs),
    ]
    for name, group in regimes:
        if not group:
            continue
        g19 = compute_metrics(group, "v19")
        g27 = compute_metrics(group, "v27")
        d_mae = g27["mae"] - g19["mae"]
        st = "IMPROVED" if d_mae < 0 else "REGRESSED"
        print(f"{name:<22} | {len(group):<6} | {g19['mae']:<10.2f} | {g27['mae']:<10.2f} | {d_mae:<+10.2f} | {st}")

    print("\n" + "=" * 100)
    print(f"TOP {args.top_k} IMAGES WHERE v27 IS WORSE THAN v19 (Regressions):")
    print("=" * 100)
    print(f"{'Image ID':<22} | {'GT':<6} | {'v19 Pred':<9} | {'v19 AE':<8} | {'v27 Pred':<9} | {'v27 AE':<8} | {'Delta AE':<9}")
    print("-" * 100)
    for d in diffs[: args.top_k]:
        print(f"{d['id']:<22} | {d['gt']:<6.0f} | {d['v19_pred']:<9.1f} | {d['v19_ae']:<8.1f} | {d['v27_pred']:<9.1f} | {d['v27_ae']:<8.1f} | {d['delta_ae']:<+9.1f}")

    print("\n" + "=" * 100)
    print(f"TOP {args.top_k} IMAGES WHERE v27 IS BETTER THAN v19 (Improvements):")
    print("=" * 100)
    print(f"{'Image ID':<22} | {'GT':<6} | {'v19 Pred':<9} | {'v19 AE':<8} | {'v27 Pred':<9} | {'v27 AE':<8} | {'Delta AE':<9}")
    print("-" * 100)
    for d in diffs[-args.top_k:]:
        print(f"{d['id']:<22} | {d['gt']:<6.0f} | {d['v19_pred']:<9.1f} | {d['v19_ae']:<8.1f} | {d['v27_pred']:<9.1f} | {d['v27_ae']:<8.1f} | {d['delta_ae']:<+9.1f}")

    if args.output_md:
        out_path = Path(args.output_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# RMR-v19 vs RMR-v27 Empirical Benchmark Report\n\n")
            f.write(f"- **RMR-v19 Path**: `{v19_p}`\n")
            f.write(f"- **RMR-v27 Path**: `{v27_p}`\n")
            f.write(f"- **Evaluated Images**: {len(diffs)}\n\n")
            f.write("## Overall Metrics\n\n")
            f.write("| Metric | RMR-v19 | RMR-v27 | Delta | Status |\n")
            f.write("|---|---|---|---|---|\n")
            f.write(f"| **MAE** | {m19_all['mae']:.2f} | {m27_all['mae']:.2f} | {mae_diff:+.2f} | **{status_mae}** |\n")
            f.write(f"| **RMSE** | {m19_all['rmse']:.2f} | {m27_all['rmse']:.2f} | {rmse_diff:+.2f} | **{status_rmse}** |\n")
            f.write(f"| **WAPE (%)** | {m19_all['wape']:.2f} | {m27_all['wape']:.2f} | {wape_diff:+.2f} | - |\n")
            f.write(f"| **Pearson r** | {m19_all['r']:.4f} | {m27_all['r']:.4f} | {r_diff:+.4f} | - |\n\n")
            f.write("## Density Regime Breakdown\n\n")
            f.write("| Regime | N | v19 MAE | v27 MAE | Delta | Status |\n")
            f.write("|---|---|---|---|---|---|\n")
            for name, group in regimes:
                if not group:
                    continue
                g19 = compute_metrics(group, "v19")
                g27 = compute_metrics(group, "v27")
                d_mae = g27["mae"] - g19["mae"]
                st = "IMPROVED" if d_mae < 0 else "REGRESSED"
                f.write(f"| {name} | {len(group)} | {g19['mae']:.2f} | {g27['mae']:.2f} | {d_mae:+.2f} | {st} |\n")
            f.write("\n")
        print(f"\n[INFO] Saved markdown report to {out_path}")


if __name__ == "__main__":
    main()
