# MICF: Measure-Consistent Integral Count Fields for Ultra-Lightweight Crowd Counting

[![Tests](https://img.shields.io/badge/pytest-13%2F13%20passed-brightgreen.svg)]()
[![Branch](https://img.shields.io/badge/branch-MICF-blue.svg)]()
[![Carrier](https://img.shields.io/badge/carrier-MobileNetV4--0.50%20(0.10M%20params)-orange.svg)]()

> **Research Hypothesis:** Directly predicting a spatial cumulative counting measure $\hat{C}(x, y)$ may provide a smoother, globally structured regression target that capacity-limited crowd counters learn more efficiently than sparse, high-frequency local counts $\hat{Y}(x, y)$—until the non-local spatial dependency requirement exceeds their representational capacity.

---

## 1. Mathematical Formulation

Let points $\mathcal{P} = \{(x_n, y_n)\}_{n=1}^N$ be raw 2D annotations on an image of size $H \times W$.

### 1.1 Exact Local Count Map (Zero Gaussian Smoothing)
For output stride $s$ (e.g. $s = 16$), the discrete local cell count map $Y \in \mathbb{N}_0^{H_o \times W_o}$ is constructed deterministically:
$$Y_{ij} = \#\left\{ n : \left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j \right\}, \qquad \sum_{i,j} Y_{ij} = N.$$

### 1.2 Cumulative Count Field
The ground-truth top-left (TL) cumulative count field $C \in \mathbb{R}_{\ge 0}^{H_o \times W_o}$ is:
$$C_{ij} = \sum_{a \le i} \sum_{b \le j} Y_{ab} = \operatorname{CumSum}_y(\operatorname{CumSum}_x(Y)).$$

In linear algebra notation with lower-triangular summation matrices $T_H, T_W$:
$$C = T_H Y T_W^\top \iff \operatorname{vec}(C) = (T_W \otimes T_H) \operatorname{vec}(Y) = P y.$$

Because $P$ is invertible with $D = T^{-1}$, **MICF adds no information** ($Y \leftrightarrow C$ is a bijection). Its contribution is a transformation of **target geometry, optimization geometry, and spatial inductive bias**.

### 1.3 Exact Inversion via Discrete Mixed Differences ($\Delta_{xy}$)
The recovered discrete mass map $\hat{Y}$ is reconstructed from $\hat{C}$ via the 2D finite difference operator:
$$\hat{Y}_{ij} = \Delta_{xy} \hat{C}_{ij} = \hat{C}_{ij} - \hat{C}_{i-1,j} - \hat{C}_{i,j-1} + \hat{C}_{i-1,j-1},$$
with zero-boundary conditions $\hat{C}_{0,j} = \hat{C}_{i,0} = 0$.

### 1.4 Measure Consistency Constraint
For $\hat{C}$ to define a valid non-negative counting measure:
$$\boxed{\Delta_{xy} \hat{C}_{ij} \ge 0 \quad \forall (i, j).}$$

### 1.5 Arbitrary Rectangle Count Recovery
For any axis-aligned bounding box $R = (x_1, x_2] \times (y_1, y_2]$:
$$N(R) = C(y_2, x_2) - C(y_1, x_2) - C(y_2, x_1) + C(y_1, x_1).$$
Total scene count is read directly from the bottom-right corner: $\hat{N}_{corner} = \hat{C}_{H_o, W_o}$.

---

## 2. Scientific Controls: The Triangle Kill-Test Suite (B1–B7)

To rigorously decouple **loss geometry** from **output representation**, the benchmark tests 7 strictly controlled variants:

```
              ┌─────────────────────────────────────────────────┐
              │             Triangle Kill-Test Suite             │
              └─────────────────────────────────────────────────┘
                     /                       \
        Local Representation (Y)       Cumulative Representation (C)
        ├── B1: SmoothL1(Y_hat, Y)     ├── B3: SmoothL1(C_hat, C) [Naive]
        ├── B2: SmoothL1(PY_hat, PY)   ├── B4: B3 + Validity Penalty
        └── B6: B1 + Integral Context  ├── B5: Full MICF-v2 (4-Dir Context)
                                       └── B7: MICF-v2 Axial (1D Context)
```

| ID | Name | Output | Supervision / Loss | Context Module | Purpose |
|:---|:---|:---:|:---|:---:|:---|
| **B1** | Local Baseline | $\hat{Y}$ | $\operatorname{SmoothL1}(\hat{Y}, Y)$ | None (Local only) | Standard local count regression benchmark |
| **B2** | Integral Loss on Local | $\hat{Y}$ | $\operatorname{SmoothL1}(P\hat{Y}, PY)$ | None (Local only) | **Isolates loss geometry**: cumulative loss without cumulative output |
| **B3** | Direct MICF (Naive) | $\hat{C}$ | $\operatorname{SmoothL1}(\hat{C}, C)$ | None | **Isolates representation**: cumulative output without validity penalty |
| **B4** | Direct MICF + Validity | $\hat{C}$ | $L_{\text{field}} + 1.0 \cdot L_{\text{valid}}$ | None | Measures impact of measure consistency constraint |
| **B5** | **Full MICF-v2** | $\hat{C}$ | $L_{\text{field}} + 1.0 \cdot L_{\text{valid}}$ | 4-Dir Integral Context | **Proposed method**: aligned 4-direction feature context + valid cumulative field |
| **B6** | Reviewer Control | $\hat{Y}$ | $\operatorname{SmoothL1}(\hat{Y}, Y)$ | 4-Dir Integral Context | Tests whether Integral Context helps local prediction independently |
| **B7** | MICF-v2 Axial | $\hat{C}$ | $L_{\text{field}} + 1.0 \cdot L_{\text{valid}}$ | Axial Integral Context | **Cheaper context** (sec.31): 1D row/col prefix averages at -2k params / -0.5 MMAC |

### Decision Logic (Kill Rules)
- **$C > B > A$**: Direct cumulative representation hypothesis survives.
- **$B \approx C > A$**: Direct cumulative output is redundant; cumulative loss is the active ingredient.
- **$B > C$**: Cumulative supervision helps, but direct cumulative prediction is bottlenecked by non-local context.
- **$A \ge B, C$**: **Kill** the integral-domain crowd counting hypothesis entirely.
- **$B7 \approx B5$**: The lightweight 1D axial context is sufficient, saving 2D multi-orientation FLOPs.

---

## 3. Architecture: MICF-Lite & Integral Context

```
Input Image [B, 3, H, W]
  │
  ▼
MobileNetV4-Conv-Small-0.50 (truncated C16, ~88k params, pretrained=True)
  │
  ▼
Additive FPN Neck (32 channels, context dilations {1, 2, 3})
  │  ├── Native multi-scale feature routes: P4 (stride 4), P8 (stride 8), P16 (stride 16)
  │  └── Direct route selection matching output_stride (zero redundant downsampling)
  │
  ▼
4-Directional Normalized Integral Context Block (Optional, B5 & B6; or Axial B7)
  │  ├── F_bar^TL = sum_{a<=i, b<=j} F_{ab} / ((i+1)(j+1))
  │  ├── F_bar^TR = sum_{a<=i, b>=j} F_{ab} / ((i+1)(W-j))
  │  ├── F_bar^BL = sum_{a>=i, b<=j} F_{ab} / ((H-i)(j+1))
  │  ├── F_bar^BR = sum_{a>=i, b>=j} F_{ab} / ((H-i)(W-j))
  │  └── Fusion: Conv1x1(5C->C) -> DW-Conv3x3 -> Conv1x1 + Residual(F)
  │      (Axial B7 uses 1D row/col prefix averages with 3C->C fusion: -2k params / -0.5 MMAC)
  │
  ▼
Task Prediction Head
  ├── Head 'local'            -> softplus(z) >= 0 (B1, B2, B6)
  ├── Head 'cumulative'       -> raw linear z (B3, B4, B5, B7)
  └── Head 'integrated_local' -> softplus(z) -> CumSum2D (Valid-by-construction control)
```

- **Parameters:** ~0.092M (B1: 92,049) vs ~0.097M (B7: 96,561) vs ~0.099M (B5: 98,609) — strictly capacity-matched.
- **Computational Cost:** ~0.065–0.066 GMACs (64.7–66.3 MMACs) at $256 \times 256$ input resolution.
- **Parallelism:** Prefix summations run in $O(HW)$ via native GPU parallel prefix scans (`torch.cumsum`). Zero learnable parameters in the pooling operator itself.

---

## 4. Losses & Optimization

### 4.1 Loss Function
$$\mathcal{L}_{\text{MICF}} = \mathcal{L}_{\text{field}}(\hat{C}, C) + \lambda_v \mathcal{L}_{\text{valid}}(\Delta_{xy} \hat{C}) + \lambda_y \mathcal{L}_{\text{local-recon}}(\Delta_{xy}\hat{C}, Y),$$
where:
- $\mathcal{L}_{\text{field}} = \operatorname{SmoothL1}(\hat{C}, C)$ over all prefix cells.
- $\mathcal{L}_{\text{valid}} = \frac{1}{HW} \sum_{i,j} \operatorname{ReLU}(-\Delta_{xy} \hat{C}_{ij})$ penalizes negative count density.
- $\mathcal{L}_{\text{local-recon}} = \operatorname{SmoothL1}(\Delta_{xy} \hat{C}, Y)$ (optional auxiliary reconstruction, $\lambda_y \in \{0, 0.01, 0.05\}$).

### 4.2 Orientation-Balanced Augmentation
Top-left cumulative supervision naturally weights cells near $(0, 0)$ more heavily because they participate in more prefixes. To eliminate this positional bias:
1. Horizontal flip is applied at random in the dataset loader ($p = 0.5$).
2. **Independent vertical flip** is applied in the training loop ($p = 0.5$) with point $y \leftarrow (H - 1) - y$.
3. Exact local $Y$ and cumulative $C$ targets are generated dynamically from the flipped points.
By symmetry:
$$(H-x)(W-y) + (H-x)y + x(W-y) + xy = HW = \text{constant},$$
guaranteeing equal expected gradient contribution across all spatial locations.

### 4.3 Training Schedule
- **Optimizer:** AdamW, initial base LR $10^{-4}$, backbone LR scale $0.1$, weight decay $10^{-4}$.
- **Schedule:** 25-epoch linear warmup followed by cosine annealing over 1000 epochs.
- **Gradient Clipping:** Max norm $5.0$.

---

## 5. Measure Diagnostics & Evaluation Regimes

### 5.1 Decoupled Evaluation Regimes (Sections 29 & 40)
To avoid confounding representation geometry with receptive-field capacity:
- **Regime A (Crop-level MAE):** Evaluates models on fixed $256 \times 256$ crops (matching training crop size). This matches the training spatial extent and reduces receptive-field distribution shift, isolating the pure representation hypothesis: does $I \to \hat{C}$ train better than $I \to \hat{Y}$?
- **Regime B (Full-image MAE):** Evaluates full uncropped images. Supports both **Direct Forward** ($MAE_{\text{full-direct}}$) and **Hierarchical Tile Composition** ($MAE_{\text{full-tiled}}$, Section 30) across all local and cumulative variants for rigorous across-regime comparison.

### 5.2 Measure Validity Diagnostics
- Negative-cell fraction: $f_- = \frac{\#\{\hat{Y}_{ij} < 0\}}{HW}$.
- Negative-mass ratio: $r_- = \frac{\sum [-\hat{Y}]_+}{\sum |\hat{Y}| + \epsilon}$.
- Violation magnitude: $V = \frac{1}{HW} \sum_{i,j} [-\hat{Y}_{ij}]_+ = \operatorname{mean}(\operatorname{ReLU}(-\hat{Y}))$.

### 5.3 Multi-Scale Rectangle Count Evaluation
Evaluates count recovery error $N(R) = C(y_2, x_2) - C(y_1, x_2) - C(y_2, x_1) + C(y_1, x_1)$ across normalized area fractions:
$$\{1/64, \; 1/16, \; 1/4, \; 1.0\}.$$

### 5.4 2D Fourier Spectral Energy Analysis
- $E_{\text{high}}$: fraction of 2D real-FFT power at spatial frequency $\|\omega\| > \tau$.
- Energy retention quantiles: coefficient fractions capturing 90%, 95%, 99% spectral energy.

---

## 6. Directory Structure

```text
lightweightcrcn/
├── MICF_full_method_design.md          # Complete mathematical & experimental design document
├── configs/
│   ├── generate_pilot_configs.py       # Config generator for B1-B7 pilot suite
│   ├── generate_capacity_sweep.py      # Config generator for neck-width capacity sweep (sec. 24)
│   ├── pilot_micf/                     # Pilot suite configs (B1-B7)
│   │   ├── b1.yaml                     # Local Count Baseline
│   │   ├── b2.yaml                     # Local Output + Integral Loss
│   │   ├── b3.yaml                     # Direct Cumulative MICF Naive (lambda_valid=0)
│   │   ├── b4.yaml                     # Direct Cumulative MICF + Validity (lambda_valid=1.0)
│   │   ├── b5.yaml                     # Full MICF-v2 (4-Dir Context + Validity)
│   │   ├── b6.yaml                     # Local Count + 4-Dir Context Control
│   │   └── b7.yaml                     # MICF-v2 Axial (1D Row/Col Context + Validity)
│   └── capacity_sweep/                 # Capacity sweep configs (b1_w16..b5_w64)
├── hpc/
│   ├── models/
│   │   ├── integral_context.py         # 4-Directional & Axial Normalized Integral Context Modules
│   │   └── micf_lite.py                # Unified MICFLite model (local, cumulative, integrated_local)
│   │                                   # + compose_tiled_cumulative_field for exact tile stitching
│   ├── losses/
│   │   └── micf.py                     # discrete_mixed_difference, cell_counts_to_cumulative_field,
│   │                                   # points_to_count_map, MICFLoss, IntegralLossOnLocalCount
│   └── diagnostics/
│       └── micf_diagnostics.py         # Measure diagnostics, rectangle queries, 2D FFT spectral analysis
├── tools/
│   ├── train_micf_pilot.py             # Authoritative trainer with orientation balancing, warmup, & Regime A/B eval
│   ├── eval_micf_comprehensive.py      # Benchmark evaluator with hierarchical tile composition (Section 50 schema)
│   └── architecture_table.py           # FLOP & parameter profiler per component
└── tests/
    └── test_micf.py                    # 13 unit tests covering all MICF math, native strides, tiling, losses, diagnostics
```

---

## 7. Quickstart & Verification

### 7.1 Run Full Test Suite
```powershell
.venv\Scripts\pytest tests/ -v
```
*(All 13 tests pass in ~6.5s).*

### 7.2 Run 1-Epoch Smoke Test (All 7 Models)
```powershell
for ($i = 1; $i -le 7; $i++) {
    .venv\Scripts\python tools/train_micf_pilot.py --config configs/pilot_micf/b$i.yaml --smoke-test
}
```

### 7.3 Train a Full Pilot Model (e.g. B5 Full MICF-v2)
```powershell
.venv\Scripts\python tools/train_micf_pilot.py --config configs/pilot_micf/b5.yaml --epochs 100
```

### 7.4 Comprehensive Benchmark Evaluation
```powershell
.venv\Scripts\python tools/eval_micf_comprehensive.py `
    --config configs/pilot_micf/b5.yaml `
    --checkpoint runs/pilot_micf/b5/best.pt `
    --output-csv ./runs/pilot_micf/benchmark_results.csv
```

Outputs the complete 20-column CSV:
```text
dataset,seed,variant,params,flops,rf_proxy,mae,mae_full_direct,mae_full_tiled,
rmse,nae,prefix_mae,local_recon_mae,rectangle_mae_small,rectangle_mae_medium,
rectangle_mae_large,negative_cell_fraction,negative_mass_ratio,violation_magnitude,peak_vram_mb
```

### 7.5 Run Capacity Sweep Config Generation
```powershell
.venv\Scripts\python configs/generate_capacity_sweep.py
```

---

## 8. Scientific Claim Boundaries

To ensure scientific defensibility:
- **Do not claim cumulative maps are novel in machine learning**: Integral images and neural CDFs have a long history. The claim is specifically: *controlled study of directly predicted spatial cumulative counting measures for ultra-lightweight crowd counting under severe capacity constraints*.
- **Do not claim cumulative fields contain "more information"**: $C = T Y T^\top$ is an invertible linear isomorphism ($Y = \Delta_{xy} C$). It contains identical Shannon information.
- **Frame as a fundamental trade-off**: Integrating the target smooths optimization geometry and encodes regional sums naturally, but induces a non-local dependency structure that challenges capacity-limited receptive fields. This trade-off is the central empirical question.
