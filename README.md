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

---

## NTPC — Neural Tree-Pólya Crowd Counting (Paper Ablations)

> **Important**: NTPC matched ablations (R0–R5) use a **separate trainer** (`train_ntpc.py`)
> and a dedicated loss module (`hpc/losses/ntpc.py`).  Do **not** use `train.py` for NTPC
> experiments — it runs the legacy HNB/HardNegative/Allocation objective which is an
> entirely different experimental setup.

### NTPC Ablation Modes

| Mode | Description |
|------|-------------|
| `r0_exact` | **Baseline**: Root-NB + mean regional L1 (64/32/16) |
| `r1_deterministic` | Root-NB + deterministic allocation (proportion matching) |
| `r2_flat_dm` | Root-NB + flat Dirichlet-Multinomial over all level-16 cells |
| `r3_multinomial_tree` | Root-NB + tree Multinomial (no concentration parameter) |
| `r4_dtm_tree16` | Root-NB + full DTM tree down to stride-16 **(proposed core)** |
| `r4_dtm_tree8` | R4 + DTM supervision at stride-8 (depth study) |
| `r4_dtm_tree4` | R4 + DTM supervision at stride-4 (depth study) |
| `r5_full_ntpc` | R4 + dense-gate 16→8 auxiliary term **(full NTPC)** |

All modes share the same Root-NB for count magnitude; spatial mechanism is the only variable.

### Run NTPC Ablation (SHA Part A, R4)

```powershell
f:\lightweightcrcn\.venv\Scripts\python train_ntpc.py --config configs/ntpc_sha.yaml
```

A minimal config template (`configs/ntpc_sha.yaml`):

```yaml
experiment:
  seed: 42
  save_dir: ./runs/ntpc_r4_sha

dataset:
  name: sha
  root: /path/to/ShanghaiTech
  crop_size: 256        # must be divisible by 64 for NTPC pyramid

augmentation:
  scale_range: [0.7, 1.3]
  flip_prob: 0.5

training:
  batch_size: 16
  epochs: 300
  amp: true
  evaluate_every: 5
  gradient_audit_every: 50

loss:
  mode: r4_dtm_tree16   # change to r0_exact … r5_full_ntpc for ablations
  root_loss: nb
  kappa_shared: 20.0

optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 5.0

schedule:
  epochs: 300
  warmup_epochs: 25

model:
  backbone: mobilenetv4_conv_small_050
  neck_width: 32
  truncate_backbone: true
```

### Target Pipeline

NTPC training generates ground-truth count pyramids via `hpc/data/point_counts.py`:

```
Y4 (H/4 × W/4)  ← rasterize points at stride-4
  → Y8  = sum_2x2(Y4)
  → Y16 = sum_2x2(Y8)
  → Y32 = sum_2x2(Y16)
  → Y64 = sum_2x2(Y32)
```

Each level is verified to be integer-valued and to conserve total count.  The `gt_blocks` dict
exposed to `NTPCLoss` always contains all five levels `{4, 8, 16, 32, 64}` regardless of mode.
The criterion selects which levels to supervise.

### Tests

```powershell
# NTPC-specific tests
f:\lightweightcrcn\.venv\Scripts\pytest tests/test_ntpc_behavior.py tests/test_point_counts.py tests/test_tree_loss.py -v

# Full suite (excluding legacy SR48 audit)
f:\lightweightcrcn\.venv\Scripts\pytest tests/ --ignore=tests/test_sr48.py -v
```
