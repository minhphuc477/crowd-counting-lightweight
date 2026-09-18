# RMR-v23 Benchmark & Ablation Study Summary

| Architecture / Variant | Epochs | Val MAE | Val RMSE | Sparse | Dense | TTA MAE | $\Delta$MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMR-v23 Canonical (BB-1 + Gated Diffusion)** | 1000 | 79.48 | 114.68 | 44.83 | 135.95 | - | - |
| **Ablation 1: No Barzilai-Borwein (Fixed omega)** | 1000 | 81.29 | 125.08 | 38.41 | 149.50 | - | +1.81 |
| **Ablation 2: No Density-Gated Diffusion** | 1000 | 81.32 | 121.77 | 43.58 | 143.24 | - | +1.84 |
| **Ablation 3: No Density-Gated Curvature** | - | - | - | - | - | - | - |
| **Ablation 4: No Scale Alignment Loss** | - | - | - | - | - | - | - |
| **Control: Direct Feedforward (No Solver)** | 1000 | 96.07 | 153.17 | 45.58 | 181.03 | - | +16.59 |
