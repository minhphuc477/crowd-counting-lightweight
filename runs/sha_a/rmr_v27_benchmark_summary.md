# RMR-v27 Benchmark & Comprehensive Ablation Suite Summary

**Target Evaluation**: ShanghaiTech Part A Test Set (182 canonical images)
**Gold Anchor**: RMR-v19 Canonical Isotropic (Direct MAE: 72.84, TTA MAE: 72.61)


| Run ID | Description | MAE | RMSE | TTA MAE | Delta v19 TTA | Sparse | Moderate | Dense | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| `rmr_v27_canonical_restored` | RMR-v27 Canonical Restored (v19 Winning Baseline: w=1.0, BB [0.2, 2.0], Curv=0.50, ScaleAlign=0.05, HardBG=0.15) | **79.43** | 118.83 | 78.52 | +5.91 | 43.89 | 58.37 | 143.91 | 1000 |
| `rmr_v27_sota_push` | RMR-v27 SOTA Push (Cyclic BB=2 + Morozov gamma=0.50 + Curv=0.35 -> Sub-70 MAE Target) | **80.34** | 124.89 | 79.79 | +7.18 | 38.99 | 57.49 | 150.72 | 1000 |
| `rmr_v27_ablation_cyclic_bb` | Ablation 1: Cyclic BB-1 (Cycle Length = 2) | **80.36** | 125.39 | 80.30 | +7.69 | 36.14 | 59.10 | 146.71 | 1000 |
| `rmr_v27_ablation_no_bb` | Ablation 2: Fixed Step Size (No BB, w=1.0) | **83.18** | 125.96 | 82.16 | +9.55 | 47.37 | 60.82 | 151.33 | 1000 |
| `rmr_v27_ablation_morozov05` | Ablation 3: Tighter Morozov Deadband (gamma=0.50) | **79.40** | 120.09 | 78.37 | +5.76 | 42.09 | 60.67 | 137.59 | 1000 |
| `rmr_v27_ablation_no_morozov` | Ablation 4: No Morozov Regularization (gamma=0.0) | **85.99** | 147.23 | 84.67 | +12.06 | 36.73 | 56.80 | 175.34 | 1000 |
| `rmr_v27_ablation_curv035` | Ablation 5: Calibrated Anscombe Curvature (lambda_curv=0.35) | **80.08** | 125.03 | - | - | 33.72 | 57.69 | 149.91 | 1000 |
| `rmr_v27_ablation_no_curvature` | Ablation 6: No Curvature Loss (Anscombe Variance-Stabilization Isolation, lambda_curv=0.0) | **-** | - | - | - | - | - | - | 4 |
| `rmr_v27_ablation_no_scale_align` | Ablation 7: No Physical Scale Alignment (lambda_scale_align=0.0) | **-** | - | - | - | - | - | - | 4 |
| `rmr_v27_ablation_no_hard_bg` | Ablation 8: No Hard Background Mining (lambda_hard_bg=0.0) | **-** | - | - | - | - | - | - | 3 |
| `rmr_v27_control_no_solver` | Control: Direct Feedforward (T=0, No SIRT Inverse Solver) | **-** | - | - | - | - | - | - | - |
