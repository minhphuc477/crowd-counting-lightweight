import csv
import json
from pathlib import Path
import yaml

runs_dir = Path("runs/canonical")
all_dirs = list(runs_dir.glob("*")) + list((runs_dir / "ablations").glob("*"))

results = []

for d in all_dirs:
    if not d.is_dir() or d.name == "ablations":
        continue
    summary_file = None
    for cand in [d / "eval_val" / "summary.json", d / "summary.json", d / "eval" / "summary.json"]:
        if cand.is_file():
            summary_file = cand
            break

    best_val_mae = None
    best_epoch = None
    last_epoch = None
    train_log = d / "train_log.csv"
    if train_log.is_file():
        try:
            with open(train_log, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_epoch = rows[-1].get("epoch")
                    val_maes = []
                    for r in rows:
                        v = r.get("val_mae") or r.get("val_direct_mae")
                        if v and v != "None":
                            try:
                                val_maes.append((float(v), int(r.get("epoch", 0))))
                            except ValueError:
                                pass
                    if val_maes:
                        val_maes.sort(key=lambda x: x[0])
                        best_val_mae, best_epoch = val_maes[0]
        except Exception:
            pass

    summary_data = {}
    if summary_file and summary_file.is_file():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    cfg_file = d / "resolved_config.yaml"
    cfg = {}
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    m_cfg = cfg.get("model", {})
    l_cfg = cfg.get("loss", {})

    results.append({
        "name": d.relative_to(runs_dir).as_posix(),
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "summary_mae": summary_data.get("mae") or summary_data.get("mae_practical"),
        "summary_rmse": summary_data.get("rmse") or summary_data.get("rmse_practical"),
        "summary_game0": summary_data.get("game0"),
        "summary_game3": summary_data.get("game3"),
        "neck": m_cfg.get("neck_type", "N/A"),
        "stride": m_cfg.get("output_stride", 4),
        "carrier": m_cfg.get("fine_carrier_type", m_cfg.get("head_type", "std")),
        "solver": m_cfg.get("enable_solver", False),
        "solver_mode": m_cfg.get("solver_mode", "direct"),
        "loss_type": l_cfg.get("allocation_loss_type", "N/A"),
        "lambda_rot": l_cfg.get("lambda_radon_ot", 0.0),
        "lambda_bay": l_cfg.get("lambda_bayesian", 0.0),
        "lambda_reg": l_cfg.get("lambda_region_nb", 0.0),
    })

results.sort(key=lambda x: (x["best_val_mae"] is None, x["best_val_mae"] or 999))

print("=" * 140)
print(f"{'Run Name':<42} | {'Best MAE':<9} | {'Ep':<4} | {'Last':<4} | {'Stride':<6} | {'Neck':<13} | {'Carrier':<10} | {'Solver':<10} | {'Loss Type'}")
print("-" * 140)
for r in results:
    b_mae = f"{r['best_val_mae']:.2f}" if r['best_val_mae'] is not None else (f"{r['summary_mae']:.2f}" if r['summary_mae'] is not None else 'N/A')
    b_ep = str(r['best_epoch']) if r['best_epoch'] is not None else '-'
    l_ep = str(r['last_epoch']) if r['last_epoch'] is not None else '-'
    solv = f"{r['solver_mode']}" if r['solver'] else "none"
    print(f"{r['name']:<42} | {b_mae:<9} | {b_ep:<4} | {l_ep:<4} | {r['stride']:<6} | {str(r['neck'])[:13]:<13} | {str(r['carrier'])[:10]:<10} | {solv:<10} | {r['loss_type']}")
print("=" * 140)
