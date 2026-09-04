"""Paired mechanism analysis: B5b vs B8 per-image AE decomposition.

Reads comprehensive_per_image.csv from both models and reports:

Statistical:
  paired bootstrap 95% CI of delta_MAE
  % images where B8 wins (AE_B8 < AE_B5b)
  median paired improvement
  mean and median per-image delta

Mechanistic decomposition:
  improvement by num_windows  (proxy for image scale)
  improvement by image area H*W
  improvement by crowd count N
  improvement by density N/(H*W)
  corr(AE_B5b - AE_B8, num_windows)  <-- key mechanistic evidence

Output: console report + paired_mechanism_report.json + 4-panel plot (PNG)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_per_image(csv_path: str) -> list[dict[str, Any]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _float_col(rows: list[dict], key: str) -> np.ndarray:
    """Extract a float column, substituting nan for missing/unparseable values."""
    out = []
    for row in rows:
        v = row.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return np.array(out, dtype=np.float64)


def _int_col(rows: list[dict], key: str) -> np.ndarray:
    out = []
    for row in rows:
        v = row.get(key, "")
        try:
            out.append(int(float(v)))
        except (ValueError, TypeError):
            out.append(0)
    return np.array(out, dtype=np.int64)


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI for the mean of x."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_means = np.array([rng.choice(x, size=n, replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(x.mean()), lo, hi


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _group_mae(ae_b8: np.ndarray, ae_b5b: np.ndarray, groupvar: np.ndarray, bins: int = 5) -> dict:
    """Bin groupvar and compute mean AE for each model per bin."""
    finite = np.isfinite(ae_b8) & np.isfinite(ae_b5b) & np.isfinite(groupvar)
    gv = groupvar[finite]
    ab8 = ae_b8[finite]
    ab5b = ae_b5b[finite]
    edges = np.percentile(gv, np.linspace(0, 100, bins + 1))
    edges[-1] += 1e-9  # include max
    result = {}
    for i in range(bins):
        mask = (gv >= edges[i]) & (gv < edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_label = f"[{edges[i]:.1f},{edges[i+1]:.1f})"
        result[bin_label] = {
            "n": int(mask.sum()),
            "mae_b8": float(ab8[mask].mean()),
            "mae_b5b": float(ab5b[mask].mean()),
            "delta_mae": float(ab5b[mask].mean() - ab8[mask].mean()),
        }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired B5b vs B8 mechanism analysis."
    )
    p.add_argument(
        "--b5b-csv",
        default="runs/pilot_micf/b5b/eval_comprehensive/comprehensive_per_image.csv",
    )
    p.add_argument(
        "--b8-csv",
        default="runs/pilot_micf/b8_k4/eval_comprehensive/comprehensive_per_image.csv",
    )
    p.add_argument("--output-dir", default="runs/pilot_micf/paired_analysis")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib figure (for headless environments).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PAIRED MECHANISM ANALYSIS: B5b vs B8")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Load CSVs
    # -----------------------------------------------------------------------
    print(f"B5b CSV : {args.b5b_csv}")
    print(f"B8  CSV : {args.b8_csv}")

    if not Path(args.b5b_csv).exists():
        raise FileNotFoundError(f"B5b CSV not found: {args.b5b_csv}")
    if not Path(args.b8_csv).exists():
        raise FileNotFoundError(f"B8 CSV not found: {args.b8_csv}")

    rows_b5b = _load_per_image(args.b5b_csv)
    rows_b8 = _load_per_image(args.b8_csv)

    # Align by image_index
    b5b_by_idx = {int(float(r["image_index"])): r for r in rows_b5b}
    b8_by_idx = {int(float(r["image_index"])): r for r in rows_b8}

    common_idxs = sorted(set(b5b_by_idx.keys()) & set(b8_by_idx.keys()))
    n = len(common_idxs)
    print(f"Matched images: {n}  (B5b: {len(rows_b5b)}, B8: {len(rows_b8)})")

    if n == 0:
        raise ValueError("No matching image indices found between the two CSVs.")

    # Extract aligned arrays
    rows_b5b_aligned = [b5b_by_idx[i] for i in common_idxs]
    rows_b8_aligned = [b8_by_idx[i] for i in common_idxs]

    ae_b5b = _float_col(rows_b5b_aligned, "err_full_direct_abs")
    ae_b8 = _float_col(rows_b8_aligned, "err_full_direct_abs")

    # Also get tiled (controlled) AE if available
    ae_b5b_tiled = _float_col(rows_b5b_aligned, "err_full_tiled_controlled_abs")
    ae_b8_tiled = _float_col(rows_b8_aligned, "err_full_tiled_controlled_abs")
    has_tiled = np.all(np.isfinite(ae_b5b_tiled)) and np.all(np.isfinite(ae_b8_tiled))

    gt_count = _float_col(rows_b5b_aligned, "gt_count")
    height = _float_col(rows_b5b_aligned, "height")
    width = _float_col(rows_b5b_aligned, "width")
    num_windows = _int_col(rows_b5b_aligned, "num_windows")

    area = height * width
    density = np.where(area > 0, gt_count / area, float("nan"))

    # delta > 0 means B5b is WORSE than B8 (B8 wins)
    delta_direct = ae_b5b - ae_b8
    delta_tiled = ae_b5b_tiled - ae_b8_tiled if has_tiled else None

    # -----------------------------------------------------------------------
    # Statistical summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY (Direct prediction)")
    print("=" * 80)

    mean_b5b = float(np.nanmean(ae_b5b))
    mean_b8 = float(np.nanmean(ae_b8))
    print(f"MAE B5b (direct):   {mean_b5b:.4f}")
    print(f"MAE B8  (direct):   {mean_b8:.4f}")
    print(f"Paired delta MAE:   {mean_b5b - mean_b8:.4f}  (positive = B8 wins)")

    delta_mean, delta_lo, delta_hi = _bootstrap_mean_ci(
        delta_direct[np.isfinite(delta_direct)],
        n_boot=args.n_bootstrap, ci=args.ci
    )
    print(f"Bootstrap {int(100*args.ci)}% CI of delta_MAE: [{delta_lo:.4f}, {delta_hi:.4f}]")

    b8_wins = np.sum(delta_direct > 0)
    b5b_wins = np.sum(delta_direct < 0)
    ties = np.sum(delta_direct == 0)
    pct_b8_wins = 100 * b8_wins / n
    print(f"B8 wins on {b8_wins}/{n} images ({pct_b8_wins:.1f}%)")
    print(f"B5b wins on {b5b_wins}/{n} images ({100*b5b_wins/n:.1f}%)")
    if ties:
        print(f"Ties: {ties}")

    median_delta = float(np.nanmedian(delta_direct))
    print(f"Median paired improvement (B8 over B5b): {median_delta:.4f}")

    if has_tiled:
        print(f"\nTiled (halo=0) MAE B5b: {np.nanmean(ae_b5b_tiled):.4f}")
        print(f"Tiled (halo=0) MAE B8:  {np.nanmean(ae_b8_tiled):.4f}")

    # -----------------------------------------------------------------------
    # Key mechanistic: corr(delta, num_windows)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MECHANISTIC CORRELATION: corr(AE_B5b - AE_B8, covariate)")
    print("=" * 80)
    print("(Positive correlation = covariate predicts B8 winning more)")

    correlations = {
        "num_windows": _pearson_r(delta_direct, num_windows.astype(float)),
        "image_area_HxW": _pearson_r(delta_direct, area),
        "gt_count_N": _pearson_r(delta_direct, gt_count),
        "density_N_per_HW": _pearson_r(delta_direct, density),
    }

    for name, r in correlations.items():
        bar = "#" * int(abs(r) * 20) if not math.isnan(r) else ""
        direction = "(+)" if (not math.isnan(r) and r > 0) else "(-)" if (not math.isnan(r)) else "(?)"
        print(f"  {name:<26}: r = {r:+.4f}  {direction} {bar}")

    r_windows = correlations["num_windows"]
    print()
    if not math.isnan(r_windows):
        if r_windows > 0.3:
            print(f"corr(delta, num_windows) = {r_windows:.4f} > 0.3")
            print("-> Strong mechanistic evidence: B8's FH representation helps on larger/multi-window images.")
        elif r_windows > 0.1:
            print(f"corr(delta, num_windows) = {r_windows:.4f} (weak positive)")
            print("-> Weak trend: FH advantage grows slightly with image scale.")
        elif r_windows < -0.1:
            print(f"corr(delta, num_windows) = {r_windows:.4f} (negative)")
            print("-> Unexpected: B5b outperforms B8 more on larger images.")
        else:
            print(f"corr(delta, num_windows) = {r_windows:.4f} (near zero)")
            print("-> No scale-dependent advantage for B8; improvement is uniform or random.")

    # -----------------------------------------------------------------------
    # Binned analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("IMPROVEMENT BY IMAGE SCALE (num_windows quintiles)")
    print("=" * 80)

    bins_windows = _group_mae(ae_b8, ae_b5b, num_windows.astype(float))
    print(f"{'Bin (num_windows)':<24} {'N':>5} {'MAE_B5b':>10} {'MAE_B8':>10} {'Delta':>10}")
    print("-" * 62)
    for bin_label, stats in bins_windows.items():
        print(f"{bin_label:<24} {stats['n']:>5} {stats['mae_b5b']:>10.2f} {stats['mae_b8']:>10.2f} {stats['delta_mae']:>+10.2f}")

    print("\n" + "=" * 80)
    print("IMPROVEMENT BY DENSITY (GT count / area, quintiles)")
    print("=" * 80)
    bins_density = _group_mae(ae_b8, ae_b5b, density)
    print(f"{'Bin (density)':<28} {'N':>5} {'MAE_B5b':>10} {'MAE_B8':>10} {'Delta':>10}")
    print("-" * 66)
    for bin_label, stats in bins_density.items():
        print(f"{bin_label:<28} {stats['n']:>5} {stats['mae_b5b']:>10.2f} {stats['mae_b8']:>10.2f} {stats['delta_mae']:>+10.2f}")

    print("\n" + "=" * 80)
    print("IMPROVEMENT BY CROWD COUNT N (quintiles)")
    print("=" * 80)
    bins_count = _group_mae(ae_b8, ae_b5b, gt_count)
    print(f"{'Bin (gt_count)':<28} {'N':>5} {'MAE_B5b':>10} {'MAE_B8':>10} {'Delta':>10}")
    print("-" * 66)
    for bin_label, stats in bins_count.items():
        print(f"{bin_label:<28} {stats['n']:>5} {stats['mae_b5b']:>10.2f} {stats['mae_b8']:>10.2f} {stats['delta_mae']:>+10.2f}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    plot_path_str = str(output_dir / "paired_mechanism_4panel.png")
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Paired Mechanism Analysis: B5b vs B8 (Direct MAE)", fontsize=14)

            # Panel 1: scatter AE_B5b vs AE_B8 colored by num_windows
            ax = axes[0, 0]
            sc = ax.scatter(ae_b5b, ae_b8, c=num_windows, cmap="viridis", alpha=0.6, s=20)
            lim = max(float(np.nanmax(ae_b5b)), float(np.nanmax(ae_b8))) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=1, label="Equal")
            ax.set_xlabel("AE B5b (Direct)")
            ax.set_ylabel("AE B8 (Direct)")
            ax.set_title("Per-image AE (color = num_windows)")
            plt.colorbar(sc, ax=ax, label="num_windows")
            ax.legend()

            # Panel 2: delta vs num_windows
            ax = axes[0, 1]
            ax.scatter(num_windows, delta_direct, alpha=0.5, s=20, c="steelblue")
            ax.axhline(0, color="k", lw=1)
            m, b = np.polyfit(num_windows.astype(float), delta_direct, 1)
            xs = np.linspace(num_windows.min(), num_windows.max(), 100)
            ax.plot(xs, m * xs + b, "r-", lw=2, label=f"r={r_windows:.3f}")
            ax.set_xlabel("num_windows")
            ax.set_ylabel("delta AE (B5b - B8)")
            ax.set_title("delta vs image scale")
            ax.legend()

            # Panel 3: delta vs gt_count
            ax = axes[1, 0]
            r_count = _pearson_r(delta_direct, gt_count)
            ax.scatter(gt_count, delta_direct, alpha=0.5, s=20, c="darkorange")
            ax.axhline(0, color="k", lw=1)
            m2, b2 = np.polyfit(gt_count[np.isfinite(gt_count)], delta_direct[np.isfinite(gt_count)], 1)
            xs2 = np.linspace(gt_count.min(), gt_count.max(), 100)
            ax.plot(xs2, m2 * xs2 + b2, "r-", lw=2, label=f"r={r_count:.3f}")
            ax.set_xlabel("GT count N")
            ax.set_ylabel("delta AE (B5b - B8)")
            ax.set_title("delta vs crowd count")
            ax.legend()

            # Panel 4: histogram of delta
            ax = axes[1, 1]
            finite_delta = delta_direct[np.isfinite(delta_direct)]
            ax.hist(finite_delta, bins=30, color="seagreen", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="k", lw=1, linestyle="--")
            ax.axvline(float(np.mean(finite_delta)), color="red", lw=2, label=f"mean={np.mean(finite_delta):.1f}")
            ax.axvline(float(np.median(finite_delta)), color="orange", lw=2, linestyle="--", label=f"median={np.median(finite_delta):.1f}")
            ax.set_xlabel("delta AE (B5b - B8), positive = B8 wins")
            ax.set_ylabel("count")
            ax.set_title(f"Distribution of paired delta (B8 wins {pct_b8_wins:.0f}%)")
            ax.legend()

            plt.tight_layout()
            plt.savefig(plot_path_str, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\nPlot saved -> {plot_path_str}")
        except ImportError:
            print("[warn] matplotlib not available, skipping plot.")
        except Exception as e:
            print(f"[warn] Plot failed: {e}")
    else:
        print("[--no-plot] Skipping figure generation.")

    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    report = {
        "b5b_csv": args.b5b_csv,
        "b8_csv": args.b8_csv,
        "n_matched_images": n,
        "direct": {
            "mae_b5b": mean_b5b,
            "mae_b8": mean_b8,
            "delta_mae_b5b_minus_b8": mean_b5b - mean_b8,
            f"bootstrap_{int(100*args.ci)}pct_ci_lower": delta_lo,
            f"bootstrap_{int(100*args.ci)}pct_ci_upper": delta_hi,
            "pct_images_b8_wins": pct_b8_wins,
            "pct_images_b5b_wins": 100 * b5b_wins / n,
            "median_delta_b5b_minus_b8": median_delta,
        },
        "correlations": correlations,
        "bins_by_num_windows": bins_windows,
        "bins_by_density": bins_density,
        "bins_by_gt_count": bins_count,
    }
    if has_tiled:
        report["tiled_controlled"] = {
            "mae_b5b": float(np.nanmean(ae_b5b_tiled)),
            "mae_b8": float(np.nanmean(ae_b8_tiled)),
            "delta_mae": float(np.nanmean(ae_b5b_tiled) - np.nanmean(ae_b8_tiled)),
        }

    report_path = output_dir / "paired_mechanism_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
