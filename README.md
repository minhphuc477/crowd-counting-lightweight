# RMR-Count: Regional Measure Regularization for Ultra-Lightweight Crowd Counting

[![Tests](https://img.shields.io/badge/pytest-100%2F100%20passed-brightgreen.svg)]()
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Model](https://img.shields.io/badge/carrier-MobileNetV4--0.50%20(82k%20params)-orange.svg)]()
[![Venue](https://img.shields.io/badge/target-CVPR%202026-purple.svg)]()

> **Core Research Hypothesis:** Capacity-constrained crowd counters (< 100k parameters) suffer from severe receptive field starvation, causing local density over-prediction in clutters and under-prediction in sparse backgrounds. Decoupling spatial count estimation into a **fine local density head** and a **multi-scale regional extensivity head**, reconciled at runtime by an unrolled implicit solver via the **exact geometric adjoint operator** $A^\top$, provides global spatial regularization without the parameter or latency cost of attention mechanisms.

---

## 1. Key Mathematical Contribution: The Adjoint Theorem

Let $Y \in \mathbb{R}_+^G$ be the discrete cell count map on spatial lattice $G$ (stride $s=4$). Let $\mathcal{R} = \{R_m\}_{m=1}^M$ be a multi-scale regional dictionary across scales $K \in \{32, 64, 128\}$ px with $50\%$ stride overlap.

### 1.1 Regional Operators
- **Forward Regional Projection Matrix** $A \in \{0, 1\}^{M \times G}$:
  $$(AY)_m = \sum_{g \in R_m} Y_g = q_m.$$
- **Adjoint Feedback Scatter Matrix** $A^\top \in \{0, 1\}^{G \times M}$:
  $$(A^\top r)_g = \sum_{m: g \in R_m} r_m.$$

### 1.2 The Adjoint Scale Invariance Theorem ($H \mathbf{1} = \mathbf{1}$)
Let $D_a = \operatorname{diag}(A \mathbf{1}_G) \in \mathbb{R}^{M \times M}$ be regional areas, and $D_c = \operatorname{diag}(A^\top \mathbf{1}_M) \in \mathbb{R}^{G \times G}$ be cell coverage counts.  
The normalized regional transfer operator is defined as:
$$H = D_c^{-1} A^\top D_a^{-1} A.$$

$$\boxed{H \mathbf{1}_G = \mathbf{1}_G \quad \forall \; \mathcal{R} \text{ covering } G.}$$

**Theoretical Significance:** Uniform crowd distributions are invariant fixed points of the feedback loop. Regional error back-projection is perfectly scale-balanced across multi-scale partitions, eliminating scale-dependent gradient explosion.

---

## 2. Architecture & Unrolled Solver

```
Input Image [1, 3, H, W]
       │
       ▼
MobileNetV4-Conv-Small-0.50 (truncated C16, ~65k params)
       │
       ▼
Additive FPN Neck (32 channels, context dilations {1, 2, 3})
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Density Head                               Regional Extensivity Head
Conv3x3 -> GN -> SiLU -> Conv1x1                ROI-Pooling on {32, 64, 128} px
init: b_0 ≈ softplus(-4.595) ≈ 0.01             + 4D Geometry [log h, log w, log A, log(w/h)]
       │                                        init: b_R = |R| * rho_R (calibrated to 0.01)
       ▼                                               │
Initial Latent Field z_0                               │
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
               RMR Implicit Unrolled Solver (T=2)
                 │
                 ├── Prefix2D fast regional integration: q = A Y^(t)
                 ├── Compute rate residual: r = (q - b) / D_a
                 ├── Adjoint back-projection: r_field = D_c^(-1) A^T r
                 ├── Latent update: z^(t+1) = z^(t) - eta_t * M^(t) * r_field
                 └── Re-activation: Y^(t+1) = softplus(z^(t+1))
                               │
                               ▼
                     Final Density Field Y_T
```

---

## 3. Registered Experimental Matrix (B0–B5)

The benchmark strictly isolates loss geometry, architectural capacity, and operator validity:

| ID | Variant Name | Params | Regional Head | Iterative Solver | Operator Property | Purpose |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 64,581 | ✗ | ✗ | None | Standard direct regression control |
| **B1** | Region Loss | 64,581 | ✗ | ✗ | Multi-scale loss on $AY$ | Auxiliary regional supervision without dual head |
| **B2** | Region Aux | 82,143 | ✓ | ✗ | Passive multi-task | Dual-head feature sharing without runtime solver |
| **B3a** | Local Refine | 88,412 | ✗ | ✓ ($T=2$) | Unconstrained GRU | Unconstrained recurrent latent refinement |
| **B3b** | Learned Projector | 94,820 | ✓ | ✓ ($T=2$) | Blackbox MLP | Unconstrained neural projection vs exact adjoint $A^\top$ |
| **B4** | RMR-T1 | 82,143 | ✓ | ✓ ($T=1$) | $H\mathbf{1}=\mathbf{1}$ | Single-step RMR projection |
| **B5** | **RMR-T2 (Registered)** | **82,143** | ✓ | ✓ ($T=2$) | $H\mathbf{1}=\mathbf{1}$ | Full registered RMR-Count model |

---

## 4. Quickstart Guide

### 4.1 Setup
```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e .
```

### 4.2 Data Manifest Generation
Generate portable manifests with relative paths and coordinate alignment:
```powershell
python -m rmr_count.prepare_manifest --dataset sha --data-root data/part_A_final --out-dir data --relative-to .
```

### 4.3 Training a Model
Train RMR-Count (B5) with mixed-precision and low-RAM safety:
```powershell
python -m rmr_count.train --config configs/rmr/rmr_t2.yaml --output_dir runs/sha_a/rmr_t2_seed42
```

### 4.4 Standalone Evaluation & Mechanism Traces
Evaluate and generate `predictions.csv`, `summary.json`, `solver_trace.csv`, and `regional_trace.csv`:
```powershell
python -m rmr_count.eval --checkpoint runs/sha_a/rmr_t2_seed42/best.pt --manifest data/sha_a_test.jsonl --out_dir eval_results/rmr_t2
```

### 4.5 Latency, FPS & Peak VRAM Profiling
Measure clean single-forward peak memory and FP32 / AMP latency:
```powershell
python -m rmr_count.profile --config configs/rmr/rmr_t2.yaml --device cuda
```

### 4.6 Statistical Paired Comparison
Compute sample-level paired differences $d_i = |\hat{N}_i^A - N_i| - |\hat{N}_i^B - N_i|$ with bootstrap 95% CI and paired t-test:
```powershell
python -m rmr_count.aggregate --compare eval_results/rmr_t2/predictions.csv eval_results/direct/predictions.csv --name-a RMR_T2 --name-b Direct
```

---

## 5. Specification Documents

Detailed technical specifications are maintained in `docs/rmr/`:
- [**Paper Specification (CVPR 2026)**](docs/rmr/PAPER_SPEC.md): Mathematical formulations, theorems, proofs, and experimental hypotheses.
- [**Implementation Specification**](docs/rmr/IMPLEMENTATION_SPEC.md): Operator engineering, prefix-sum caching, low-RAM data loader, and training schedule.
- [**Evaluation Specification**](docs/rmr/EVALUATION_SPEC.md): Canonical metric definitions (NAE, physical GAME), diagnostic trace schemas, and statistical significance testing.
