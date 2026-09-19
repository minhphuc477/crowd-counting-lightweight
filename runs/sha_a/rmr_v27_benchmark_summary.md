# RMR-v27 Benchmark & Comprehensive Ablation Suite Summary

**Target Evaluation**: ShanghaiTech Part A Test Set (182 canonical images)
**Gold Anchor**: RMR-v19 Canonical Isotropic (Direct MAE: 72.84, TTA MAE: 72.61)


| Run ID | Description | MAE | RMSE | TTA MAE | Delta v19 TTA | Sparse | Moderate | Dense | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| `rmr_v27_canonical_restored` | RMR-v27 Canonical Restored (v19 Winning Baseline: w=1.0, BB [0.2, 2.0], Curv=0.50, ScaleAlign=0.05, HardBG=0.15) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_sota_push` | RMR-v27 SOTA Push (Cyclic BB=2 + Morozov gamma=0.50 + Curv=0.35 -> Sub-70 MAE Target) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_cyclic_bb` | Ablation 1: Cyclic BB-1 (Cycle Length = 2) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_no_bb` | Ablation 2: Fixed Step Size (No BB, w=1.0) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_morozov05` | Ablation 3: Tighter Morozov Deadband (gamma=0.50) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_no_morozov` | Ablation 4: No Morozov Regularization (gamma=0.0) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_curv035` | Ablation 5: Calibrated Anscombe Curvature (lambda_curv=0.35) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_no_curvature` | Ablation 6: No Curvature Loss (Anscombe Variance-Stabilization Isolation, lambda_curv=0.0) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_no_scale_align` | Ablation 7: No Physical Scale Alignment (lambda_scale_align=0.0) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_ablation_no_hard_bg` | Ablation 8: No Hard Background Mining (lambda_hard_bg=0.0) | **-** | - | - | - | - | - | - | - |
| `rmr_v27_control_no_solver` | Control: Direct Feedforward (T=0, No SIRT Inverse Solver) | **-** | - | - | - | - | - | - | - |
