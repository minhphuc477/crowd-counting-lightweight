import os
import glob
import json
import yaml
import csv

versions = ["rmr_v54", "rmr_v55", "rmr_v56", "rmr_v57", "rmr_v53"]

print("=" * 80)
print("AUDIT STATUS OF EXPERIMENTAL RUNS")
print("=" * 80)

for v in versions:
    base = os.path.join("runs", "sha_a", v)
    if not os.path.exists(base):
        print(f"\n=== {v}: DIRECTORY NOT FOUND ===")
        continue
    runs = sorted(os.listdir(base))
    print(f"\n=== {v} ({len(runs)} items) ===")
    for r in runs:
        r_dir = os.path.join(base, r)
        if not os.path.isdir(r_dir):
            continue
        log_csv = os.path.join(r_dir, "train_log.csv")
        summary_json = os.path.join(r_dir, "eval_val", "summary.json")
        last_pt = os.path.join(r_dir, "last.pt")
        best_pt = os.path.join(r_dir, "best_val_mae.pt")
        cfg_file = os.path.join(r_dir, "resolved_config.yaml")

        total_epochs = "N/A"
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r") as f:
                    cfg = yaml.safe_load(f)
                    total_epochs = cfg.get("training", {}).get("epochs", 1000)
            except Exception:
                pass

        epochs_done = 0
        best_mae_in_log = None
        if os.path.exists(log_csv):
            try:
                with open(log_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    maes = []
                    for row in reader:
                        epochs_done += 1
                        if "val_mae" in row and row["val_mae"]:
                            try:
                                maes.append(float(row["val_mae"]))
                            except ValueError:
                                pass
                    if maes:
                        best_mae_in_log = min(maes)
            except Exception as e:
                epochs_done = -1

        status = "UNKNOWN"
        if total_epochs != "N/A":
            if epochs_done >= total_epochs:
                status = f"FINISHED ({epochs_done}/{total_epochs} ep)"
            else:
                status = f"INCOMPLETE ({epochs_done}/{total_epochs} ep) -> NEED RESUME"
        else:
            status = f"{epochs_done} epochs"

        eval_info = "No final eval"
        if os.path.exists(summary_json):
            try:
                with open(summary_json) as f:
                    s = json.load(f)
                mae = s.get("mae", s.get("val_mae", "N/A"))
                rmse = s.get("rmse", s.get("val_rmse", "N/A"))
                if isinstance(mae, (int, float)):
                    eval_info = f"Final Eval: MAE={mae:.2f}, RMSE={rmse:.2f}"
                else:
                    eval_info = f"Final Eval: MAE={mae}"
            except Exception as e:
                eval_info = f"Eval error: {e}"

        log_mae_info = f"BestLogMAE={best_mae_in_log:.2f}" if best_mae_in_log is not None else ""

        print(f"  [{r}]")
        print(f"     Status:   {status}")
        print(f"     Ckpts:    last.pt={'YES' if os.path.exists(last_pt) else 'NO'}, best.pt={'YES' if os.path.exists(best_pt) else 'NO'}")
        print(f"     Metrics:  {log_mae_info} | {eval_info}")
