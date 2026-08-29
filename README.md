# NTPC: Neural Tree-Pólya Crowd Counting for Robust Lightweight Deployment

## Overview

**NTPC** (Neural Tree-Pólya Crowd Counter) is a single-map lightweight crowd counting and localization framework engineered for robust real-world deployment across sparse, dense, dark, blurry, and empty/negative scenes (such as ShanghaiTech, UCF-QNRF, and NWPU-Crowd).

### Key Architectural Principles
- **Minimal Deployed Graph**:
  $$\text{Image} \xrightarrow{} \text{MobileNetV4-Conv-Small-0.5} \xrightarrow{} \text{Additive 32-ch FPN Neck} \xrightarrow{} 1\text{-channel Softplus Mass Map } D \xrightarrow{} \hat{C} = \sum D$$
  - Deployed parameters: ~0.35M parameters (with physical truncation of reduction-32 stages).
  - Single-scale, single-head feed-forward inference at stride 4 predicting a count-mass map \(D\).
  - No Gaussian density targets, transformer decoders, or Hungarian matching in the deployed counting graph. (Hungarian matching is used only for offline localization evaluation).

### Training-Only Structured Objectives (R0–R5)
All modes share the same Root-NB for count magnitude ($\text{Var}(N) = \mu + \mu^2 / r$); spatial supervision is the only variable.

| Mode | Description |
|------|-------------|
| `r0_exact` | **Baseline**: Root-NB + mean regional L1 ($\frac{L_{64}+L_{32}+L_{16}}{3}$) |
| `r1_deterministic` | Root-NB + deterministic allocation (proportion matching) |
| `r2_flat_dm` | Root-NB + flat Dirichlet-Multinomial over all level-16 cells |
| `r3_multinomial_tree` | Root-NB + tree Multinomial (no concentration parameter) |
| `r4_dtm_tree16` | Root-NB + full DTM tree down to stride-16 **(proposed core)** |
| `r4_dtm_tree8` | R4 + DTM supervision down to stride-8 (depth study) |
| `r4_dtm_tree4` | R4 + DTM supervision down to stride-4 (depth study) |
| `r5_full_ntpc` | R4 + dense-gate 16→8 auxiliary term **(optional adaptive extension)** |

---

## Directory Structure

```
lightweightcrcn/
├── README.md
├── requirements.txt
├── pyproject.toml
├── train_ntpc.py                # Single authoritative NTPC trainer (R0-R5)
├── evaluate.py                  # Standalone full-image counting evaluation
├── configs/
│   ├── ntpc_sha.yaml            # Standard SHA default config
│   ├── ntpc_r0_exact_regression.yaml
│   ├── ntpc_r1_sdc_deterministic.yaml
│   ├── ntpc_r2_flat_dm16.yaml
│   ├── ntpc_r3_hierarchical_multinomial.yaml
│   ├── ntpc_r4_neural_dtm_tree.yaml
│   └── ntpc_r5_full_adaptive_ntpc.yaml
├── hpc/
│   ├── models/
│   │   ├── backbone.py          # MobileNetV4 with reduction-16 truncation
│   │   ├── blocks.py            # ConvGNAct, DepthwiseDilated, DSResidual, RepDWBlock
│   │   ├── neck.py              # 32-ch additive FPN with multi-dilation context
│   │   ├── factory.py           # Unified model builder & checkpoint compatibility validator
│   │   └── hpc_lite.py          # HPCLite model & data-driven head bias initialization
│   ├── losses/
│   │   ├── negative_binomial.py # Root NB and Poisson likelihoods
│   │   └── ntpc.py              # NTPCLoss supporting R0-R5 objectives
│   ├── data/
│   │   ├── transforms.py        # NTPCGeometricTransform (scale, crop, flip, exact coords)
│   │   ├── common.py            # BaseCrowdDataset with exact count pyramids
│   │   ├── point_counts.py      # Recursive count tree: Y4 -> Y8 -> Y16 -> Y32 -> Y64
│   │   ├── sha.py               # ShanghaiTech Part A / Part B loader
│   │   ├── qnrf.py              # UCF-QNRF dataset loader
│   │   ├── nwpu.py              # NWPU-Crowd dataset loader
│   │   └── sampler.py           # Density & luminance balanced sampler
│   ├── evaluation/
│   │   └── counting.py          # Shared evaluate_counting protocol
│   ├── metrics/
│   │   ├── counting.py          # MAE, RMSE, NAE, Bias metrics
│   │   ├── localization.py      # Hungarian matching, F1@4px, F1@8px
│   │   ├── otm.py               # Optimal Transport Monge parameter-free localizer
│   │   └── subgroup.py          # Diagnostic bins & luminance evaluation
│   └── utils/
│       └── seed.py              # Deterministic seeding & worker initialization
├── tools/
│   ├── eval_localization.py     # Joint counting & OT-M localization evaluator
│   ├── eval_ntpc_localization_depth.py # Hierarchy depth evaluator
│   ├── profile_model.py         # Latency, MACs, FPS, parameter breakdown
│   ├── export_onnx.py           # ONNX export & dynamic multi-shape parity check
│   ├── visualize_localization.py# Side-by-side localization visualizer
│   ├── summary_runs.py          # Multi-run evaluation summary and decision check
│   └── create_smoke_dataset.py  # Synthetic crowd dataset generator for smoke tests
└── tests/                       # NTPC unit and integration test suite
    ├── test_ntpc_math.py        # Dirichlet-Multinomial math & tree collapse identity
    ├── test_ntpc_targets.py     # Recursive integer count pyramids & validation
    ├── test_ntpc_geometry.py    # Geometry transforms & coordinate invariance
    ├── test_ntpc_loss.py        # Loss behaviors, scale invariance, gradient checks
    ├── test_ntpc_model.py       # Architecture shapes, padding invariance & parameter budget
    ├── test_ntpc_overfit.py     # 1-image and batch optimization sanity tests
    ├── test_localization.py     # Hungarian point matching & F1 metrics
    ├── test_otm_official.py     # OT-M official Monge solver tests
    └── test_dtm4_otm.py         # Stride-4 DTM tree & OT-M integration
```

---

## Quickstart

### 1. Run Unit & Integration Tests
```powershell
.venv\Scripts\pytest tests/ -v
```

### 2. Train NTPC Model (R4 Core on ShanghaiTech Part A)
```powershell
.venv\Scripts\python train_ntpc.py --config configs/ntpc_r4_neural_dtm_tree.yaml
```

### 3. Evaluate Checkpoint
```powershell
.venv\Scripts\python evaluate.py --config configs/ntpc_r4_neural_dtm_tree.yaml --checkpoint runs/ntpc_r4_neural_dtm_tree/best.pt
```

### 4. Evaluate OT-M Localization
```powershell
.venv\Scripts\python tools/eval_localization.py --config configs/ntpc_r4_neural_dtm_tree.yaml --checkpoint runs/ntpc_r4_neural_dtm_tree/best.pt --output runs/ntpc_r4_neural_dtm_tree/loc.json
```

### 5. Profile Model Efficiency & Parameters
```powershell
.venv\Scripts\python tools/profile_model.py --config configs/ntpc_sha.yaml
```

### 6. Export to ONNX & Verify Dynamic Shapes
```powershell
.venv\Scripts\python tools/export_onnx.py --config configs/ntpc_sha.yaml --checkpoint runs/ntpc_r4_neural_dtm_tree/best.pt --output hpc_lite.onnx
```

---

## Target Pipeline

Ground-truth count pyramids are generated deterministically in `hpc/data/point_counts.py`:

```
Y4 (H/4 × W/4)  ← rasterize points at stride-4: floor((x + 0.5)/4), floor((y + 0.5)/4)
  → Y8  = sum_2x2(Y4)
  → Y16 = sum_2x2(Y8)
  → Y32 = sum_2x2(Y16)
  → Y64 = sum_2x2(Y32)
  → N   = sum(Y64)
```

- Target consistency is validated at runtime: all count cells are exact non-negative integers, and every parent cell strictly equals the sum of its $2\times 2$ child cells.
- The `is_exact_joint_nll` property on `NTPCLoss` verifies that the active configuration corresponds to the true factorized joint negative log-likelihood.
