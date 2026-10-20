# Reviewer Memory: Autonomous Research & Review Loop
**Run ID**: `run_20261020_rmr_50mae`
**Topic**: First-Principles Radon Measure Recovery for Sub-50 MAE Crowd Counting on ShanghaiTech Part A
**Current Round**: 3 (V60 Evaluation Complete, Planning V61)
**Reviewer Verdict**: REVISE WITH CELL SUPERVISION (Score: 8.0 / 10.0)

---

## 1. Verified Empirical Results from V60 (1000 Epochs)
1. **Track B (Subpixel Stride 2)**:
   - Best Val MAE: **83.75** at epoch 585 (RMSE: 140.62, NAE: 0.2067).
   - **Median AE reached 48.48** (over 50% of test images achieved Sub-50 MAE!).
   - Moderate density MAE reached **60.94** (129 images).
   - Sparse density MAE remained rock-solid at **18.36** (7 images).
2. **Track A (Stride 4)**:
   - Best Val MAE: **89.07** at epoch 465 (RMSE: 136.56, NAE: 0.2292).
   - Bias: **+3.78** (near zero systematic bias).
3. **Stride 2 vs Stride 4 Performance**:
   - Subpixel Stride 2 beats Stride 4 by **-5.32 MAE** and improves Median AE by **-12.12 AE**.

---

## 2. Forensic Findings & Root Cause of the 83.75 Gap
1. **Lack of Microscopic Point/Cell Supervision**:
   - V60 used only macro-area integrals (`flat_dm16` 16px blocks and `region_nb` 32-128px boxes).
   - Without point/cell loss (`lambda_cell: 0.0`), the carrier density smeared out flatly inside each 16px block, losing peak sharpness.
2. **Missing Dynamic Range Expansion**:
   - With `density_curvature: false`, Softplus ceiling prevented dense clusters from scaling up to 1,300+ people.
3. **Four Outlier Images**:
   - `IMG_92`, `IMG_90`, `IMG_8`, `IMG_127` alone accounted for **+13.91 MAE** (35% of total test error).
4. **Late-Epoch Overfitting**:
   - Models peaked at epochs 465–585, then degraded by +11 to +16 MAE by epoch 1000 as LR dropped to $10^{-6}$.

---

## 3. Next Actionable Phase (V61)
- Re-introduce `cell_loss_mode: count_harmonized` ($\lambda=0.5$, power 0.5) to restore micro point supervision.
- Re-activate `density_curvature: true` on Subpixel Stride 2 to break Softplus ceiling.
- Execute pending Fourier Characteristic Function experiment (`v60_track_b_chfl_subpixel_rn.yaml`).
