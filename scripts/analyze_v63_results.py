import json
from pathlib import Path

runs_dir = Path("runs/sha_a/rmr_v63")
results = []

for r_dir in sorted(runs_dir.iterdir()):
    if not r_dir.is_dir():
        continue
    summ_path = r_dir / "eval_val" / "summary.json"
    if summ_path.is_file():
        with open(summ_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        results.append({
            "name": r_dir.name,
            "mae": d.get("mae", d.get("MAE")),
            "rmse": d.get("rmse", d.get("RMSE")),
            "nae": d.get("nae", d.get("NAE")),
            "sparse_mae": d.get("mae_sparse"),
            "moderate_mae": d.get("mae_moderate"),
            "dense_mae": d.get("mae_dense"),
            "game0": d.get("GAME0"),
            "game1": d.get("GAME1"),
            "game2": d.get("GAME2"),
            "game3": d.get("GAME3"),
            "bias": d.get("bias", d.get("Bias")),
            "p90ae": d.get("P90AE"),
        })

print(f"{'Run Name':<42} | {'MAE':<6} | {'RMSE':<6} | {'Sparse':<6} | {'Mod':<6} | {'Dense':<6} | {'GAME3':<6} | {'NAE':<6} | {'Bias':<6}")
print("-" * 115)
for r in results:
    s_mae = f"{r['sparse_mae']:.2f}" if r['sparse_mae'] is not None else "N/A"
    m_mae = f"{r['moderate_mae']:.2f}" if r['moderate_mae'] is not None else "N/A"
    d_mae = f"{r['dense_mae']:.2f}" if r['dense_mae'] is not None else "N/A"
    g3 = f"{r['game3']:.2f}" if r['game3'] is not None else "N/A"
    print(
        f"{r['name']:<42} | "
        f"{r['mae']:<6.2f} | "
        f"{r['rmse']:<6.2f} | "
        f"{s_mae:<6} | "
        f"{m_mae:<6} | "
        f"{d_mae:<6} | "
        f"{g3:<6} | "
        f"{r['nae']:<6.4f} | "
        f"{r['bias']:<6.2f}"
    )
