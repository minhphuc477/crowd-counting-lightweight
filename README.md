# RMR-Count: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

[![Tests](https://img.shields.io/badge/pytest-100%2F100%20passed-brightgreen.svg)]()
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/carrier-64.5k%20params-orange.svg)]()
[![Target](https://img.shields.io/badge/venue-CVPR%202026-purple.svg)]()

> **Core Research Question:** In ultra-lightweight crowd counting (< 100k parameters), maintaining both high-resolution local spatial fidelity and long-range spatial consistency is challenging under strict mobile computation budgets. Because learning dense global self-attention or deep multi-scale dilated receptive fields is parameter-prohibitive, we explore:  
> $$\boxed{\textbf{Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?}}$$  
>  
> **Scope:** RMR-Count is an explicit counting and spatial density estimation model. Point localization, detection bounding boxes, and Hungarian matching are outside the scope of this work.

---

## 1. Key Mathematical Contribution: The Adjoint Theorem

Let $Y \in \mathbb{R}_+^G$ be the discrete cell count map on spatial lattice $G$ (stride $s=4$). The canonical ground truth per discrete cell is:
$$Y_{ij}^* = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left(\left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j\right),$$
with cell coordinates clamped to valid feature grid bounds $[0, H_o - 1] \times [0, W_o - 1]$.

Let $\mathcal{R} = \{R_m\}_{m=1}^M$ be a multi-scale regional dictionary across scales $K \in \{32, 64, 128\}$ px with $50\%$ stride overlap.

### 1.1 Discrete Regional Operators
- **Forward Regional Projection Matrix** $A \in \{0, 1\}^{M \times G}$:
  $$(AY)_m = \sum_{g \in R_m} Y_g = q_m.$$
- **Adjoint Back-Projection Matrix** $A^\top \in \{0, 1\}^{G \times M}$:
  $$(A^\top r)_g = \sum_{m: g \in R_m} r_m.$$

### 1.2 The Adjoint Scale Invariance Theorem ($H \mathbf{1} = \mathbf{1}$)
Let $D_a = \operatorname{diag}(A \mathbf{1}_G) \in \mathbb{R}^{M \times M}$ be regional areas, and $D_c = \operatorname{diag}(A^\top \mathbf{1}_M) \in \mathbb{R}^{G \times G}$ be cell coverage counts.  
The normalized regional transfer operator is defined as:
$$H = D_c^{-1} A^\top D_a^{-1} A.$$

$$\boxed{H \mathbf{1}_G = \mathbf{1}_G \quad \forall \; \mathcal{R} \text{ covering } G.}$$

**Theoretical Significance:** Uniform crowd distributions are invariant fixed points of the feedback loop. Regional error back-projection is scale-balanced across multi-scale partitions, preventing scale-dependent gradient explosion or boundary artifacts.

---

## 2. Architecture & Operator-Guided Reconciliation

The framework uses an ultra-lightweight custom convolutional backbone and neck, decoupling fine local density prediction from regional extensivity estimation, followed by explicit unrolled reconciliation:

```
Input Image [1, 3, H, W]
       │
       ▼
TinyLocalEncoder (~52k params)
  ├── Stem: ConvGNAct(3 -> 16, k=3, stride=2)
  ├── s4:   TinyIR(16 -> 24, stride=2) + TinyIR(24 -> 24)
  ├── s8:   TinyIR(24 -> 40, stride=2) + 2x TinyIR(40 -> 40)
  └── s16:  TinyIR(40 -> 64, stride=2) + TinyIR(64 -> 64)
       │
       ▼
AdditiveFusion Neck (~3.5k params, width=32)
  ├── 1x1 Projections of C4, C8, C16 to 32 ch
  ├── Bilinear upsampling to C4 resolution + elementwise addition
  └── Fused via depthwise-separable 3x3 ConvGNAct + 1x1 ConvGNAct
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Density Head (~3.2k params)               Regional Extensivity Head (~4.2k params)
Depthwise-sep Conv3x3 + Conv1x1                ROI-Pooling on {32, 64, 128} px
init bias: b_0 ≈ -4.595 (softplus ≈ 0.01)      + 4D Geometry [log h, log w, log A, log(w/h)]
       │                                        init: b_R = |R| * rho_R (calibrated to 0.01)
       ▼                                               │
Initial Latent Field z_0                               │
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            Unrolled Regional Reconciliation (T=2, ~1.5k params)
                 │
                 ├── Fast 2D Prefix Sum: q = A Y^(t)
                 ├── Rate residual: r = (q - b) / D_a
                 ├── Adjoint feedback: r_field = D_c^(-1) A^T r
                 ├── Latent update: z^(t+1) = z^(t) - eta_t * M^(t) * r_field
                 └── Re-activation: Y^(t+1) = softplus(z^(t+1))
                               │
                               ▼
                     Final Density Field Y_T
```

Step sizes are parameterized as $\eta_t = \eta_{\text{max}} \cdot \sigma(\alpha_t)$ with per-iteration learnable logits $\alpha_t$ initialized to $\eta_t(0) = 0.05$ (with $\eta_{\text{max}} = 0.20$).

---

## 3. Registered Experimental Matrix (B0–B5)

Exact parameter counts computed via `count_parameters(model)`:

| ID | Variant Name | Parameter Count | Regional Head | Unrolled Reconciliation | Exact Adjoint $A^\top$ | Scientific Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 58,867 | ✗ | ✗ | ✗ | Direct regression control without regional reasoning |
| **B1** | Region Loss | 58,867 | ✗ (Loss on $AY$) | ✗ | ✗ | Auxiliary regional rate loss without dual head |
| **B2** | Region Aux | 63,044 | ✓ | ✗ | ✗ | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 61,876 | ✗ | ✓ ($T=2$) | ✗ (Unconstrained) | Receptive field expansion via local recurrent refinement |
| **B3b** | Learned Projector | 66,086 | ✓ | ✓ ($T=2$) | ✗ (Blackbox MLP) | Learned neural projection vs exact mathematical adjoint $A^\top$ |
| **B4** | RMR-T1 | 64,580 | ✓ | ✓ ($T=1$) | ✓ ($H\mathbf{1}=\mathbf{1}$) | Single-step unrolled reconciliation |
| **B5** | **RMR-T2 (Registered)** | **64,581** | ✓ | ✓ ($T=2$) | ✓ ($H\mathbf{1}=\mathbf{1}$) | Two-step registered RMR-Count model |

---

## 4. Quickstart Guide

### 4.1 Environment Setup
```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e .
```

### 4.2 Dataset Preprocessing & Manifests
Generate portable manifests with relative image references and boundary coordinate alignment:
```powershell
python -m rmr_count.prepare_manifest `
    --images data/part_A_final/train_data/images `
    --annotations data/part_A_final/train_data/ground_truth `
    --dataset sha_a `
    --out data/sha_a_train.jsonl `
    --relative-to .
```

### 4.3 Training
Train RMR-Count (B5) using mixed precision and low-RAM safe cropping:
```powershell
python -m rmr_count.train `
    --config configs/rmr/rmr_t2.yaml `
    --seed 42 `
    --lr 3e-4 `
    --output-dir runs/sha_a/rmr_t2_seed42
```

### 4.4 Standalone Evaluation & Diagnostic Traces
Evaluate a trained model and output `predictions.csv`, `summary.json`, `solver_trace.csv`, and `regional_trace.csv`:
```powershell
python -m rmr_count.eval `
    --checkpoint runs/sha_a/rmr_t2_seed42/best_val_mae.pt `
    --manifest data/sha_a_val.jsonl `
    --out-dir eval_results/rmr_t2
```

### 4.5 Latency, FPS & Peak VRAM Profiling
Measure clean single-forward peak memory and FP32 / AMP latency:
```powershell
python -m rmr_count.profile --config configs/rmr/rmr_t2.yaml --device cuda
```

### 4.6 Statistical Paired Comparison
Compute sample-level paired differences $d_i = |\hat{N}_i^A - N_i| - |\hat{N}_i^B - N_i|$ with bootstrap 95% CI and paired t-test:
```powershell
python -m rmr_count.aggregate `
    --compare eval_results/rmr_t2/predictions.csv eval_results/direct/predictions.csv `
    --name-a RMR_T2 `
    --name-b Direct
```

---

## 5. Canonical Documentation

Detailed technical specifications are maintained in `docs/rmr/`:
- [**Paper Specification (CVPR 2026)**](docs/rmr/PAPER_SPEC.md): Theoretical derivations, proofs ($H\mathbf{1}=\mathbf{1}$), step-size formulation, and experimental hypotheses.
- [**Implementation Specification**](docs/rmr/IMPLEMENTATION_SPEC.md): Exact layer dimensions, operator caching, low-RAM data contracts, and loss variant dispatch.
- [**Evaluation Specification**](docs/rmr/EVALUATION_SPEC.md): Canonical metric definitions (NAE, physical GAME), diagnostic trace schemas, and statistical significance testing.
