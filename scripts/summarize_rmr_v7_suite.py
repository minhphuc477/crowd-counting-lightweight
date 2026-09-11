from __future__ import annotations

import argparse
import csv
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
        row = {
            "name": rd.name,
            "trained_epochs": "-",
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

        # 1. Read train_log.csv to find total trained epochs and best epoch
        log_csv = rd / "train_log.csv"
        if log_csv.exists():
            try:
                with open(log_csv, "r", encoding="utf-8") as f:
                    r = list(csv.DictReader(f))
                    if r:
                        row["trained_epochs"] = r[-1].get("epoch", "-")
                        val_rows = [x for x in r if x.get("val_mae") and x["val_mae"] != ""]
                        if val_rows:
                            best_r = min(val_rows, key=lambda x: float(x["val_mae"]))
                            row["best_epoch"] = best_r.get("epoch", "-")
            except Exception:
                pass

        # 2. Candidate summary JSONs
        summary_candidates = [
            rd / "eval_val" / "summary.json",
            rd / "summary.json",
            rd / "eval_metrics.json",
        ] + list(rd.glob("eval_*/summary.json")) + list(rd.glob("eval_*/eval_metrics.json"))

        for sp in summary_candidates:
            if sp.exists():
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    
                    # Top-level metrics
                    for src_k, target_k in [
                        ("MAE", "mae"), ("mae", "mae"),
                        ("RMSE", "rmse"), ("rmse", "rmse"),
                        ("NAE", "nae"), ("nae", "nae"),
                        ("Bias", "bias"), ("bias", "bias"),
                        ("GAME0", "game0"), ("game_0", "game0"),
                        ("GAME1", "game1"), ("game_1", "game1"),
                        ("GAME2", "game2"), ("game_2", "game2"),
                        ("GAME3", "game3"), ("game_3", "game3"),
                        ("mae_sparse", "mae_sparse"),
                        ("mae_moderate", "mae_mod"),
                        ("mae_dense", "mae_dense"),
                    ]:
                        if src_k in s and row[target_k] == "-":
                            val = float(s[src_k])
                            if target_k == "nae":
                                row[target_k] = f"{val:.3f}"
                            elif target_k == "bias":
                                row[target_k] = f"{val:+.2f}"
                            else:
                                row[target_k] = f"{val:.2f}"

                    # Nested density_stratified_mae
                    strat = s.get("density_stratified_mae", {})
                    if "<=100" in strat and row["mae_sparse"] == "-":
                        row["mae_sparse"] = f"{float(strat['<=100']['mae']):.2f}"
                    if "100-500" in strat and row["mae_mod"] == "-":
                        row["mae_mod"] = f"{float(strat['100-500']['mae']):.2f}"
                    if ">500" in strat and row["mae_dense"] == "-":
                        row["mae_dense"] = f"{float(strat['>500']['mae']):.2f}"

                    if s.get("best_epoch") and row["best_epoch"] == "-":
                        row["best_epoch"] = str(s["best_epoch"])
                except Exception:
                    pass

        rows.append(row)

    md = [
        "# RMR-v7 Benchmark Summary Table (ShanghaiTech Part A)\n\n",
        "| Model Run | Trained Epochs | Best Val Epoch | Test MAE | RMSE | NAE | Bias | GAME-0 | GAME-1 | GAME-2 | GAME-3 | Sparse (<=100) | Moderate (101-500) | Dense (>500) |\n",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]

    for r in rows:
        md.append(
            f"| **{r['name']}** | {r['trained_epochs']} | {r['best_epoch']} | **{r['mae']}** | {r['rmse']} | {r['nae']} | {r['bias']} | "
            f"{r['game0']} | {r['game1']} | {r['game2']} | {r['game3']} | {r['mae_sparse']} | {r['mae_mod']} | {r['mae_dense']} |\n"
        )

    md_content = "".join(md)
    print(md_content)

    out_p = Path(args.output_md)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(md_content, encoding="utf-8")
    print(f"\nSaved benchmark summary to {out_p}")


if __name__ == "__main__":
    main()
