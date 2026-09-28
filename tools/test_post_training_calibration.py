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


def run_eval(ckpt_path: str, use_tta: bool = False, device: str = "cuda:0"):
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
        use_tta=use_tta,
    )
    return rows, summary


def analyze_calibration(rows, name="Model"):
    gts = np.array([r["gt"] for r in rows])
    preds = np.array([r["pred"] for r in rows])
    errors = np.abs(preds - gts)
    raw_mae = float(np.mean(errors))
    raw_rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    raw_bias = float(np.mean(preds - gts))

    sparse_mask = gts <= 100
    mod_mask = (gts > 100) & (gts <= 500)
    dense_mask = gts > 500

    print(f"\n--- {name} (Raw) ---")
    print(f"MAE: {raw_mae:.2f} | RMSE: {raw_rmse:.2f} | Bias: {raw_bias:+.2f}")
    print(f"  Sparse (n={sparse_mask.sum()}): {np.mean(errors[sparse_mask]):.2f}")
    print(f"  Moderate (n={mod_mask.sum()}): {np.mean(errors[mod_mask]):.2f}")
    print(f"  Dense (n={dense_mask.sum()}):    {np.mean(errors[dense_mask]):.2f}")

    # Optimal global scalar calibration: pred_cal = s * pred
    # Find s in [0.8, 1.2]
    best_s = 1.0
    best_s_mae = raw_mae
    for s in np.linspace(0.85, 1.15, 61):
        cal_preds = s * preds
        m = float(np.mean(np.abs(cal_preds - gts)))
        if m < best_s_mae:
            best_s_mae = m
            best_s = s

    # Optimal global bias calibration: pred_cal = pred - bias
    cal_bias_preds = preds - raw_bias
    bias_mae = float(np.mean(np.abs(cal_bias_preds - gts)))

    # Optimal affine calibration: a * pred + b
    A = np.vstack([preds, np.ones_like(preds)]).T
    a, b = np.linalg.lstsq(A, gts, rcond=None)[0]
    affine_preds = a * preds + b
    affine_mae = float(np.mean(np.abs(affine_preds - gts)))

    print(f"Post-Training Calibration on {name}:")
    print(f"  Scale calibration (s={best_s:.3f}): MAE = {best_s_mae:.2f} (delta: {best_s_mae - raw_mae:+.2f})")
    print(f"  Bias subtraction (shift={-raw_bias:+.2f}): MAE = {bias_mae:.2f} (delta: {bias_mae - raw_mae:+.2f})")
    print(f"  Affine regression (a={a:.3f}, b={b:.1f}): MAE = {affine_mae:.2f} (delta: {affine_mae - raw_mae:+.2f})")

    return {
        "name": name,
        "gts": gts,
        "preds": preds,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_bias": raw_bias,
        "best_s": best_s,
        "best_s_mae": best_s_mae,
        "bias_mae": bias_mae,
        "affine_mae": affine_mae,
    }


if __name__ == "__main__":
    print("Testing m04 and m08 on test set...")
    r04, s04 = run_eval("runs/sha_a/matrix/m04_iso_fractional_loss/best_val_mae.pt", use_tta=False)
    d04 = analyze_calibration(r04, "m04_iso_fractional_loss")

    r04_tta, s04_tta = run_eval("runs/sha_a/matrix/m04_iso_fractional_loss/best_val_mae.pt", use_tta=True)
    d04_tta = analyze_calibration(r04_tta, "m04_iso_fractional_loss + TTA (HFlip)")

    r08, s08 = run_eval("runs/sha_a/matrix/m08_step3_triad_cures/best_val_mae.pt", use_tta=False)
    d08 = analyze_calibration(r08, "m08_step3_triad_cures")

    r08_tta, s08_tta = run_eval("runs/sha_a/matrix/m08_step3_triad_cures/best_val_mae.pt", use_tta=True)
    d08_tta = analyze_calibration(r08_tta, "m08_step3_triad_cures + TTA (HFlip)")

    # Test Post-Training Ensemble Blending between m04 and m08
    print("\n--- Post-Training Ensemble Blending (m04 + m08) ---")
    gts = d04["gts"]
    p04 = d04["preds"]
    p08 = d08["preds"]
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        p_blend = alpha * p04 + (1 - alpha) * p08
        mae_blend = float(np.mean(np.abs(p_blend - gts)))
        rmse_blend = float(np.sqrt(np.mean((p_blend - gts) ** 2)))
        bias_blend = float(np.mean(p_blend - gts))
        print(f"  alpha={alpha:.1f} (m04*{alpha:.1f} + m08*{1-alpha:.1f}): MAE = {mae_blend:.2f} | RMSE = {rmse_blend:.2f} | Bias = {bias_blend:+.2f}")

    p04_t = d04_tta["preds"]
    p08_t = d08_tta["preds"]
    print("\n--- Post-Training Ensemble Blending with TTA (m04_TTA + m08_TTA) ---")
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        p_blend_t = alpha * p04_t + (1 - alpha) * p08_t
        mae_blend_t = float(np.mean(np.abs(p_blend_t - gts)))
        rmse_blend_t = float(np.sqrt(np.mean((p_blend_t - gts) ** 2)))
        bias_blend_t = float(np.mean(p_blend_t - gts))
        print(f"  alpha={alpha:.1f}: MAE = {mae_blend_t:.2f} | RMSE = {rmse_blend_t:.2f} | Bias = {bias_blend_t:+.2f}")
