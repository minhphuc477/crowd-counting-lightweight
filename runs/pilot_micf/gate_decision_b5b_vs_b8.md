# Gate Decision Report: B5b vs B8 (FH-CMICF K=4)

**Dataset**: ShanghaiTech Part A (`seed=42`)
**Carrier**: MobileNetV4-Conv-Small-0.50 (99,697 parameters)
**Strict Control**: Confounder-free architecture (identical parameter count, conv RF scope, and GroupNorm spatial scope).

## Validation Trajectory Summary (Training History)

| Metric | B5b (Global Extent-Aware) | B8 (FH-CMICF K=4) | Absolute Delta (B8 - B5b) |
| :--- | :---: | :---: | :---: |
| **MAE_crop** (Best Validation) | 23.99 | 21.54 | -2.45 |
| **Val Tiled MAE** (At Best Crop Epoch) | 118.31 | 105.98 | -12.33 |
| **Violation Rate** | 1.54% | 1.43% | -0.11% |
| **Violation Magnitude** | 0.0002 | 0.0002 | +0.0001 |
| **Negative Mass Ratio** | 0.50% | 0.57% | +0.08% |

## Comprehensive Test Set Evaluation (Unbiased Protocols)

### 1. Counting Performance Across Regimes

| Evaluation Regime | Metric | B5b (Global Extent-Aware) | B8 (FH-CMICF K=4) | Absolute Delta (B8 - B5b) |
| :--- | :--- | :---: | :---: | :---: |
| **Regime A: Fixed 256x256 Validation Crop** | MAE_crop | 23.99 | 21.54 | -2.45 |
| | Val Tiled MAE | 118.31 | 105.98 | -12.33 |
| **Regime B: Controlled Tiled (tile=256, halo=0, matched extent $A_{\max}=256$)** | MAE | 110.18 | 103.23 | -6.95 |
| | RMSE | 186.81 | 184.46 | -2.35 |
| | NAE | 0.2872 | 0.2617 | -0.0255 |
| | SRE | 8.1635 | 7.4774 | -0.6861 |
| **Regime C: Practical Tiled (tile=256, halo=64)** | MAE | 118.31 | 105.98 | -12.33 |
| | RMSE | 191.32 | 184.31 | -7.01 |
| | NAE | 0.3160 | 0.2689 | -0.0471 |
| | SRE | 8.8005 | 7.5900 | -1.2104 |
| **Regime D: Full Direct (Unconstrained inference)** | MAE | 209.63 | 115.22 | -94.41 |
| | RMSE | 308.52 | 188.76 | -119.76 |
| | NAE | 0.5910 | 0.3009 | -0.2901 |
| | SRE | 15.5419 | 8.0511 | -7.4908 |

### 2. Generalization Gaps & Sensitivity Analysis

| Diagnostic Gap | Metric | B5b (Global Extent-Aware) | B8 (FH-CMICF K=4) | Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Direct - Tiled Practical Gap** | $\Delta$ MAE | +91.32 | +9.24 | -82.08 |
| **Direct - Tiled Controlled Gap** | $\Delta$ MAE | +99.45 | +11.99 | -87.46 |
| **Halo Effect (Practical - Controlled)** | $\Delta$ MAE | +8.14 | +2.75 | -5.39 |

### 3. Patch / Window & Localization (GAME) Metrics

| Metric | B5b | B8 | Delta (B8 - B5b) |
| :--- | :---: | :---: | :---: |
| **Window MAE (Micro)** | 16.35 | 13.45 | -2.91 |
| **Window MAE (Macro)** | 17.20 | 14.55 | -2.64 |
| **Empty Window MAE** | 2.57 | 2.16 | -0.41 |
| **Non-Empty Window MAE** | 18.37 | 15.10 | -3.27 |
| **Cancellation Ratio (Mean)** | 31.43% | 28.73% | -2.70% |
| **GAME(0) Tiled / Direct** | 118.46 / 199.93 | 106.34 / 115.38 | -12.12 / -84.55 |
| **GAME(1) Tiled / Direct** | 146.71 / 278.20 | 124.91 / 138.90 | -21.81 / -139.30 |
| **GAME(2) Tiled / Direct** | 176.69 / 338.97 | 150.38 / 166.42 | -26.31 / -172.56 |
| **GAME(3) Tiled / Direct** | 224.25 / 421.41 | 190.87 / 205.44 | -33.38 / -215.97 |

### 4. Measure Validity & Representation Diagnostics

| Metric | B5b (Tiled / Direct) | B8 (Tiled / Direct) |
| :--- | :---: | :---: |
| **Violation Rate (raw)** | 6.12% / 20.76% | 4.46% / 3.78% |
| **Violation Rate ($\tau=10^{-6}$)** | 6.12% / 20.76% | 4.46% / 3.78% |
| **Negative Mass Ratio** | 1.13% / 19.44% | 0.58% / 1.47% |
| **Cumulative Field NMAE** | 0.0970 / 0.1174 | 0.0892 / 0.0986 |
| **Measure Normalized L1** | 1.4871 / 3.1098 | 1.3260 / 1.3210 |


## Scientific Gate Verdict

$$\boxed{B8 >> B5b (FH factorization significantly superior -> proceed to seed expansion & K-sweep)}$$

- Trajectory curves saved to: `runs/pilot_micf/comparison_b5b_vs_b8_curves.png`
- 2D spatial error map saved to: `runs/pilot_micf/spatial_error_map_C_b5b_vs_b8.png`
