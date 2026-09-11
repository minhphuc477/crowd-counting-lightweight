from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize RMR-v7 benchmark results across all runs.")
    parser.add_argument("--runs-dir", default="runs/sha_a", help="Directory containing run directories")
    parser.add_argument("--output-md", default="runs/sha_a/rmr_v7_benchmark_summary.md", help="Path to output Markdown table")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return

    v7_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and "rmr_v7" in d.name])
    if not v7_runs:
        print(f"No rmr_v7 run directories found in {runs_dir}")
        return

    rows = []
    for rd in v7_runs:
        summary_path = rd / "summary.json"
        eval_path = rd / "eval_metrics.json"

        row = {
            "name": rd.name,
            "best_epoch": "-",
            "mae": "-",
            "rmse": "-",
            "nae": "-",
            "bias": "-",
            "game0": "-",
            "game1": "-",
            "game2": "-",
            "game3": "-",
            "mae_sparse": "-",
            "mae_mod": "-",
            "mae_dense": "-",
        }

        # Read summary.json from training
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    row["best_epoch"] = s.get("best_epoch", "-")
                    best_m = s.get("best_metrics", {})
                    for k in ("MAE", "RMSE", "NAE", "Bias", "GAME0", "GAME1", "GAME2", "GAME3"):
                        if k in best_m:
                            row[k.lower()] = f"{float(best_m[k]):.2f}" if k != "NAE" else f"{float(best_m[k]):.3f}"
                    if "mae_sparse" in best_m:
                        row["mae_sparse"] = f"{float(best_m['mae_sparse']):.2f}"
                    if "mae_moderate" in best_m:
                        row["mae_mod"] = f"{float(best_m['mae_moderate']):.2f}"
                    if "mae_dense" in best_m:
                        row["mae_dense"] = f"{float(best_m['mae_dense']):.2f}"
            except Exception:
                pass

        # If full eval_metrics.json exists (from eval.py), prefer its exact numbers
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                    row["mae"] = f"{float(ev.get('mae', row['mae'])):.2f}"
                    row["rmse"] = f"{float(ev.get('rmse', row['rmse'])):.2f}"
                    row["nae"] = f"{float(ev.get('nae', row['nae'])):.3f}"
                    row["bias"] = f"{float(ev.get('bias', row['bias'])):+.2f}"
                    for g in (0, 1, 2, 3):
                        if f"game_{g}" in ev:
                            row[f"game{g}"] = f"{float(ev[f'game_{g}']):.2f}"
                    strat = ev.get("density_stratified_mae", {})
                    if "<=100" in strat:
                        row["mae_sparse"] = f"{float(strat['<=100']['mae']):.2f}"
                    if "100-500" in strat:
                        row["mae_mod"] = f"{float(strat['100-500']['mae']):.2f}"
                    if ">500" in strat:
                        row["mae_dense"] = f"{float(strat['>500']['mae']):.2f}"
            except Exception:
                pass

        rows.append(row)

    md = [
        "# RMR-v7 Benchmark Summary Table\n\n",
        "| Model Run | Best Epoch | Test MAE | RMSE | NAE | Bias | GAME-0 | GAME-1 | GAME-2 | GAME-3 | Sparse (<=100) | Moderate (101-500) | Dense (>500) |\n",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]

    for r in rows:
        md.append(
            f"| **{r['name']}** | {r['best_epoch']} | **{r['mae']}** | {r['rmse']} | {r['nae']} | {r['bias']} | "
            f"{r['game0']} | {r['game1']} | {r['game2']} | {r['game3']} | {r['mae_sparse']} | {r['mae_mod']} | {r['mae_dense']} |\n"
        )

    md_content = "".join(md)
    print(md_content)

    out_p = Path(args.output_md)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(md_content, encoding="utf-8")
    print(f"Saved benchmark summary to {out_p}")


if __name__ == "__main__":
    main()
