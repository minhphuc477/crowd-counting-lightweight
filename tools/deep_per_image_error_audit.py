import json
import sys
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.eval import load_model_from_ckpt


def get_model_preds(ckpt_path: str, device: str = "cuda:0"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _, cfg, _ = load_model_from_ckpt(Path(ckpt_path), dev, use_ema=True)
    dataset = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=dev,
        output_stride=4,
        run_tiling=True,
        tile_size=512,
        practical_halo=64,
        use_tta=False,
    )
    return rows, summary


if __name__ == "__main__":
    checkpoints = {
        "m04_fractional": "runs/sha_a/matrix/m04_iso_fractional_loss/best_val_mae.pt",
        "m08_triad": "runs/sha_a/matrix/m08_step3_triad_cures/best_val_mae.pt",
        "champion_synth": "runs/sha_a/sub60_champion_synthesis/best_val_mae.pt",
        "triad_calib": "runs/sha_a/sub60_triad_calibrated/best_val_mae.pt",
    }

    results = {}
    for name, path in checkpoints.items():
        if Path(path).is_file():
            print(f"Evaluating {name}...")
            rows, summ = get_model_preds(path)
            results[name] = rows

    if not results:
        print("No models evaluated.")
        sys.exit(1)

    # All share the same GTs
    base_rows = results["m04_fractional"]
    gts = np.array([r["gt"] for r in base_rows])
    img_ids = [r["id"] for r in base_rows]
    total_samples = len(gts)

    bins = [
        ("Ultra-Sparse (N <= 50)", gts <= 50),
        ("Sparse (50 < N <= 150)", (gts > 50) & (gts <= 150)),
        ("Low-Medium (150 < N <= 300)", (gts > 150) & (gts <= 300)),
        ("Medium-High (300 < N <= 600)", (gts > 300) & (gts <= 600)),
        ("Dense (600 < N <= 1000)", (gts > 600) & (gts <= 1000)),
        ("Ultra-Dense (N > 1000)", gts > 1000),
    ]

    print("\n" + "=" * 105)
    print("FINE-GRAINED DENSITY TIER BREAKDOWN ACROSS MODELS")
    print("=" * 105)

    header = f"{'Density Bin':<28} | {'N':>3} | {'m04_MAE':>8} {'(Bias)':>7} | {'m08_MAE':>8} {'(Bias)':>7} | {'champ_MAE':>8} {'(Bias)':>7} | {'triad_MAE':>8} {'(Bias)':>7}"
    print(header)
    print("-" * len(header))

    for bin_name, mask in bins:
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        vals = []
        for model_name in ["m04_fractional", "m08_triad", "champion_synth", "triad_calib"]:
            p = np.array([r["pred"] for r in results[model_name]])
            err = np.abs(p[mask] - gts[mask])
            bias = p[mask] - gts[mask]
            vals.append((float(np.mean(err)), float(np.mean(bias))))
        line = f"{bin_name:<28} | {n_bin:>3} |"
        for m_err, m_bias in vals:
            line += f" {m_err:>8.2f} {m_bias:>+7.1f} |"
        print(line)

    # Total Error Mass Contribution
    print("\n" + "=" * 105)
    print("SHARE OF TOTAL ERROR MASS BY DENSITY BIN (m04_fractional)")
    print("=" * 105)
    p_m04 = np.array([r["pred"] for r in results["m04_fractional"]])
    all_errors_m04 = np.abs(p_m04 - gts)
    total_err_m04 = float(np.sum(all_errors_m04))

    for bin_name, mask in bins:
        n_bin = int(mask.sum())
        bin_err_sum = float(np.sum(all_errors_m04[mask]))
        pct = (bin_err_sum / total_err_m04) * 100
        print(f"  {bin_name:<28} (n={n_bin:>2}): sum_AE = {bin_err_sum:>8.1f} | {pct:>5.1f}% of total dataset error!")

    # Top 15 Worst Images in m04
    worst_indices = np.argsort(all_errors_m04)[::-1][:15]
    print("\n" + "=" * 105)
    print("TOP 15 WORST PREDICTION OUTLIERS IN m04 (Ranked by Absolute Error)")
    print("=" * 105)
    print(f"{'Rank':<5} | {'Image ID':<12} | {'GT':>6} | {'m04 Pred':>8} {'(Err)':>7} | {'m08 Pred':>8} {'(Err)':>7} | {'Champ Pred':>10} {'(Err)':>7}")
    print("-" * 105)
    for rank, idx in enumerate(worst_indices, 1):
        gt = gts[idx]
        img_id = img_ids[idx]
        p4 = results["m04_fractional"][idx]["pred"]
        p8 = results["m08_triad"][idx]["pred"]
        pc = results["champion_synth"][idx]["pred"]
        print(f"{rank:<5} | {img_id:<12} | {gt:>6.1f} | {p4:>8.1f} ({abs(p4-gt):>6.1f}) | {p8:>8.1f} ({abs(p8-gt):>6.1f}) | {pc:>10.1f} ({abs(pc-gt):>6.1f})")
