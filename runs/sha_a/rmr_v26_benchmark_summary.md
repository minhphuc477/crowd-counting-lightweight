# RMR-v26 Benchmark & Ablation Suite Summary

**Target Evaluation**: ShanghaiTech Part A Test Set (182 canonical images)


| Run ID | Description | MAE | RMSE | TTA MAE | Delta MAE | Sparse | Moderate | Dense | Epochs |
|---|---|---|---|---|---|---|---|---|---|
| `rmr_v26_canonical` | RMR-v26 Canonical (MPE-v2 + Pure BB-1 [0.5, 1.2]*w + Morozov gamma=0.75 + Curv=0) | **82.52** | 131.17 | 79.76 | - | 33.78 | 61.04 | 150.18 | 1000 |
| `rmr_v26_ablation_no_elevation` | Ablation 1: No Perspective Elevation (Carrier without MPE-v2) | **82.96** | 131.23 | 83.22 | +0.44 | 37.33 | 58.90 | 157.38 | 1000 |
| `rmr_v26_ablation_no_bb` | Ablation 2: No BB Step Size Damping (Fixed omega=0.20) | **79.90** | 125.15 | 77.39 | -2.62 | 29.58 | 59.71 | 144.19 | 1000 |
| `rmr_v26_ablation_no_morozov` | Ablation 3: No Morozov Regularization (gamma=0.0) | **80.91** | 126.90 | 78.96 | -1.61 | 34.47 | 59.89 | 146.91 | 1000 |
| `rmr_v26_ablation_with_curv01` | Ablation 4: With Curvature Regularization (lambda_curv=0.10) | **76.92** | 116.87 | 73.76 | -5.60 | 24.18 | 59.90 | 132.68 | 1000 |
| `rmr_v26_control_no_solver` | Control: Direct Feedforward (T=0, No SIRT Inverse Solver) | **85.92** | 134.67 | 82.61 | +3.39 | 41.14 | 66.75 | 146.48 | 1000 |