# HPC-S: Hierarchical Probabilistic Crowd Counting for Robust Lightweight Deployment

## Overview

**HPC-S** (Hierarchical Probabilistic Crowd Counter - Small Variant) is a single-map lightweight crowd counting and localization framework engineered for robust real-world deployment across sparse, dense, dark, blurry, and empty/negative scenes (such as NWPU-Crowd, ShanghaiTech, and UCF-QNRF).


### Key Architectural Principles
- **Minimal Deployed Graph**:
  $$\text{Image} \xrightarrow{} \text{MobileNetV4-Conv-Small-0.5} \xrightarrow{} \text{Additive 32-ch FPN Neck} \xrightarrow{} 1\text{-channel Softplus Mass Map } D \xrightarrow{} \hat{C} = \sum D$$
  - Deployed parameters: $< 1.5\text{M}$ (with physical truncation of reduction-32 stages).
  - Single-scale, single-head feed-forward inference at stride 4.
  - No Gaussian density maps, no transformer decoders, no Hungarian matchers, no multi-scale test ensembles.

### Training-Only Structured Objectives
1. **Hierarchical Negative-Binomial Likelihood ($L_{HNB}$)**:
   - Supervises exact integer block counts at multiple spatial scales ($B \in \{16, 32, 64, 96\}$).
   - Accounts for crowd count overdispersion ($\text{Var}(Y) = \mu + \mu^2 / r$).
   - Density-stratified block risk reduction balancing empty ($G_0$), sparse ($G_1$), medium ($G_2$), and dense ($G_3$) blocks.
2. **Block-Constrained Normalized Spatial Allocation ($L_{alloc}$)**:
   - Bilinearly splats point annotations strictly inside $16\times 16$ input pixel blocks ($4\times 4$ stride-4 cells).
   - Prevents boundary leakage across blocks, conserving integer count targets ($\sum_{k=1}^{16} Z_{bk} = Y_b^{(16)}$).
   - Normalizes by $(1/y_b)$ so every positive block contributes equal spatial supervision.
3. **Hard-Negative Zero-Block Mass Mining ($L_{HN}$)**:
   - Dynamically mines top-$\rho$ (default 10%) highest hallucinated mass blocks among true zero blocks per image.
4. **Whole-Image Empty Suppression ($L_{empty}$)**:
   - Explicitly penalizes total mass $\sum D$ on GT-empty images ($C=0$).
5. **Global Count Loss ($L_{global}$)**:
   - SmoothL1 loss between $\log(1+\hat{C})$ and $\log(1+C)$ for global conservation.
6. **Adverse-Condition Clean $\to$ Degraded Consistency ($L_{rob}$)**:
   - Supervises degraded views (brightness, contrast, blur, noise, compression) against clean detached teacher predictions.
7. **Density & Luminance Balanced Sampler**:
   - Training-only $2\text{D}$ quantile grouping with weights $w_i = 1/\sqrt{n_{g(i), l(i)}}$.

---

## Directory Structure

```
hpc_lite/
├── README.md
├── requirements.txt
├── configs/
│   ├── sha.yaml
│   ├── shb.yaml
│   ├── qnrf.yaml
│   └── nwpu.yaml
├── hpc/
│   ├── models/
│   │   ├── blocks.py            # ConvGNAct, DepthwiseDilated, DSResidual
│   │   ├── backbone.py          # MobileNetV4 with reduction-16 truncation
│   │   ├── neck.py              # 32-ch additive FPN with multi-dilation context
│   │   └── hpc_lite.py          # HPCLite model & data-driven bias initialization
│   ├── targets/
│   │   ├── block_counts.py      # Exact integer block count targets Y^(B)
│   │   └── allocation_target.py # Block-constrained soft allocation targets Z^(16)
│   ├── losses/
│   │   ├── negative_binomial.py # Stable NB NLL & stratified block risk
│   │   ├── allocation.py        # Normalized spatial allocation loss
│   │   ├── hard_negative.py     # Top-k zero-block mining & empty image loss
│   │   ├── robustness.py        # Clean -> degraded consistency loss
│   │   └── criterion.py         # Total criterion & curriculum schedule
│   ├── data/
│   │   ├── transforms.py        # Geometric & Photometric transformations
│   │   ├── common.py            # BaseCrowdDataset
│   │   ├── sha.py               # ShanghaiTech Part A / Part B
│   │   ├── qnrf.py              # UCF-QNRF dataset loader
│   │   ├── nwpu.py              # NWPU-Crowd dataset loader
│   │   └── sampler.py           # Density & luminance balanced sampler
│   ├── metrics/
│   │   ├── counting.py          # MAE, RMSE, NAE
│   │   └── subgroup.py          # NWPU diagnostic bins & luminance evaluation
│   └── utils/
│       ├── seed.py              # Deterministic seeding
│       ├── logging.py           # CSV logger
│       └── checkpoint.py        # Model checkpoint management
├── tools/
│   ├── compute_dataset_stats.py # Precompute dataset stats & dispersion init
│   ├── profile_model.py         # Latency, MACs, FPS, parameter breakdown
│   ├── export_onnx.py           # ONNX export & numerical parity check
│   └── analyze_errors.py        # Detailed diagnostic error reporting
├── train.py                     # Full training pipeline with AMP & curriculum
├── evaluate.py                  # Evaluation with subgroup & corruption benchmarks
└── tests/                       # Unit test suite (T1 - T12 & tiny-overfit)
```

---

## Quickstart

### 1. Run Unit Tests
```powershell
f:\lightweightcrcn\.venv\Scripts\pytest tests/ -v
```

### 2. Profile Model Efficiency
```powershell
f:\lightweightcrcn\.venv\Scripts\python tools/profile_model.py --config configs/sha.yaml
```

### 3. Precompute Training Statistics
```powershell
f:\lightweightcrcn\.venv\Scripts\python tools/compute_dataset_stats.py --config configs/sha.yaml --output stats_sha.json
```

### 4. Train Model
```powershell
f:\lightweightcrcn\.venv\Scripts\python train.py --config configs/sha.yaml
```

### 5. Evaluate Model
```powershell
f:\lightweightcrcn\.venv\Scripts\python evaluate.py --config configs/sha.yaml --checkpoint runs/sha/best.pt
```

### 6. Export to ONNX
```powershell
f:\lightweightcrcn\.venv\Scripts\python tools/export_onnx.py --config configs/sha.yaml --checkpoint runs/sha/best.pt --output hpc_lite.onnx
```
