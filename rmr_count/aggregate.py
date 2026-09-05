from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(
    diffs: np.ndarray,
    num_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute empirical bootstrap confidence interval for the mean of diffs."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, n, size=(num_samples, n))
    boot_means = diffs[indices].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def read_predictions_csv(path: str | Path) -> dict[str, dict[str, float]]:
    """Read predictions.csv indexed by sample id."""
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row.get("id") or row.get("image_id")
            if sample_id is None:
                continue
            data[str(sample_id)] = {
                k: float(v) for k, v in row.items() if k not in ("id", "image_id") and v != ""
            }
    return data


def compare_predictions(
    csv_a: str | Path,
    csv_b: str | Path,
    name_a: str = "method_a",
    name_b: str = "method_b",
    pred_col: str = "pred",
) -> dict:
    """Compute paired error comparison between method A and method B."""
    preds_a = read_predictions_csv(csv_a)
    preds_b = read_predictions_csv(csv_b)

    common_ids = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    if not common_ids:
        raise ValueError(f"No common image IDs found between {csv_a} and {csv_b}")

    gts = []
    pa_list = []
    pb_list = []
    ea_list = []
    eb_list = []

    for cid in common_ids:
        ra = preds_a[cid]
        rb = preds_b[cid]
        gt = ra.get("gt", ra.get("gt_count", 0.0))
        pa = ra[pred_col]
        pb = rb[pred_col]
        ea = abs(pa - gt)
        eb = abs(pb - gt)

        gts.append(gt)
        pa_list.append(pa)
        pb_list.append(pb)
        ea_list.append(ea)
        eb_list.append(eb)

    ea = np.asarray(ea_list, dtype=np.float64)
    eb = np.asarray(eb_list, dtype=np.float64)
    diff = ea - eb  # d_i = |hat_N^A - N| - |hat_N^B - N|. If < 0, A has lower error than B.

    d_mean = float(diff.mean())
    d_std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(diff)

    # Paired t-test
    try:
        ttest_res = stats.ttest_rel(ea, eb)
        p_ttest = float(ttest_res.pvalue)
    except Exception:
        p_ttest = None

    # Wilcoxon signed-rank test
    try:
        w_res = stats.wilcoxon(ea, eb)
        p_wilcoxon = float(w_res.pvalue)
    except Exception:
        p_wilcoxon = None

    wins_a = int(np.sum(ea < eb))
    wins_b = int(np.sum(eb < ea))
    ties = int(np.sum(ea == eb))

    mae_a = float(ea.mean())
    mae_b = float(eb.mean())
    rmse_a = float(np.sqrt(np.mean(ea ** 2)))
    rmse_b = float(np.sqrt(np.mean(eb ** 2)))

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_samples": len(common_ids),
        name_a: {
            "mae": mae_a,
            "rmse": rmse_a,
            "wins": wins_a,
        },
        name_b: {
            "mae": mae_b,
            "rmse": rmse_b,
            "wins": wins_b,
        },
        "ties": ties,
        "delta_mae": mae_a - mae_b,  # positive means B has lower error
        "paired_difference": {
            "definition": f"|{name_a} - GT| - |{name_b} - GT|",
            "mean": d_mean,
            "std": d_std,
            "bootstrap_95_ci": [ci_lo, ci_hi],
            "p_value_paired_ttest": p_ttest,
            "p_value_wilcoxon": p_wilcoxon,
        },
    }


def aggregate_summaries(summary_paths: list[str | Path]) -> dict:
    """Aggregate multiple summary.json files across runs."""
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in summary_paths]
    keys = sorted(set.intersection(*(set(r) for r in rows)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            a = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "min": float(a.min()),
                "max": float(a.max()),
                "n": len(a),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate summary.json metrics or compute paired bootstrap comparisons from predictions.csv."
    )
    ap.add_argument("files", nargs="*", help="Files to process (summary.json files or two predictions.csv files)")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("CSV_A", "CSV_B"), help="Two prediction CSV files to compare")
    ap.add_argument("--name-a", default="method_a", help="Name for method A in comparison")
    ap.add_argument("--name-b", default="method_b", help="Name for method B in comparison")
    ap.add_argument("--pred-col", default="pred", help="Column name for prediction (e.g. pred or pred_tiled_practical)")
    ap.add_argument("--output", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    if args.compare:
        result = compare_predictions(
            args.compare[0],
            args.compare[1],
            name_a=args.name_a,
            name_b=args.name_b,
            pred_col=args.pred_col,
        )
    elif len(args.files) == 2 and all(p.endswith(".csv") for p in args.files):
        result = compare_predictions(
            args.files[0],
            args.files[1],
            name_a=args.name_a,
            name_b=args.name_b,
            pred_col=args.pred_col,
        )
    elif args.files and all(p.endswith(".json") for p in args.files):
        result = aggregate_summaries(args.files)
    elif args.files:
        # Check first file extension
        first = Path(args.files[0])
        if first.suffix == ".json":
            result = aggregate_summaries(args.files)
        else:
            raise ValueError(f"Unrecognized file types in {args.files}")
    else:
        ap.print_help()
        return

    formatted = json.dumps(result, indent=2)
    print(formatted)
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted, encoding="utf-8")


if __name__ == "__main__":
    main()
