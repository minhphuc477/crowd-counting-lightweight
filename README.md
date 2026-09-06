# RMR: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

[![CI](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml/badge.svg)](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml)
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/carrier-101.7k%20params-orange.svg)]()
[![Target](https://img.shields.io/badge/venue-CVPR%202026-purple.svg)]()

> **Core Research Question:** In ultra-lightweight crowd counting (< 105k parameters), maintaining both fine spatial cell fidelity and long-range spatial consistency is challenging under strict mobile computation budgets. Because learning dense global self-attention or deep multi-scale dilated receptive fields is parameter-prohibitive, we explore:  
> $$\boxed{\textbf{Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?}}$$  
>  
> **Scope:** Explicit counting and spatial density estimation models. Point localization, detection bounding boxes, and Hungarian matching are outside the primary causal contribution.

---

## 1. Repository Packages & Architecture

The repository is organized into a modular, clean hierarchy:

- **`rmr_core/`**: Shared canonical primitives:
  - `backbones.py`: Dynamic MobileNetV4 backbone with reductions $\{4, 8, 16\}$ discovery.
  - `necks.py`: Additive FPN neck and inverted residual building blocks.
  - `heads.py`: Fine measure density head with empirical prior initialization.
  - `operators.py`: Discrete rectangular projection matrix $A$, exact adjoint $A^\top$, and LRU-cached multi-scale region geometry.
  - `data.py`: Manifest dataset, coordinate-exact transformations, and rasterization.
  - `metrics.py`: Canonical crowd-counting NAE, physical-support GAME with fractional cell boundary integration, and bootstrap confidence intervals.
  - `evaluation.py`: Unified evaluation engine supporting direct/tiled inference and sample identifier tracking.
  - `training.py`: Deterministic seed controls and cosine schedule builders.
- **`rmr_v2/`**: Frozen reference implementation of **RMR-Count / Stage C** (B0–B5-P).
- **`rmr_v3/`**: Active development of **Reliability-Weighted Regional Measure Reconciliation (RW-RMR / RMR-v3)**.
- **`rmr_count/`**: Backward-compatibility facade ensuring continuous execution of existing scripts and background runs.
- **`legacy/`**: Isolated historical codebases (MICF pilot, capacity sweeps, receptive-field sweeps).

---

## 2. Key Mathematical Foundations

### 2.1 Discrete Regional Operators
Let $Y \in \mathbb{R}_+^G$ be the discrete cell count measure on spatial lattice $G$ (stride $s=4$). The canonical ground truth per cell is:
$$Y_{ij}^* = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left(\left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j\right).$$

- **Forward Regional Projection** $A \in \{0, 1\}^{M \times G}$: $(AY)_m = \sum_{g \in R_m} Y_g = q_m$.
- **Adjoint Back-Projection** $A^\top \in \{0, 1\}^{G \times M}$: $(A^\top r)_g = \sum_{m: g \in R_m} r_m$.

### 2.2 The Adjoint Transfer Theorem ($H \mathbf{1} = \mathbf{1}$)
Let $D_a = \operatorname{diag}(A \mathbf{1}_G) \in \mathbb{R}^{M \times M}$ be regional areas, and $D_c = \operatorname{diag}(A^\top \mathbf{1}_M) \in \mathbb{R}^{G \times G}$ be cell coverage counts.  
The normalized regional transfer operator is defined as:
$$H = D_c^{-1} A^\top D_a^{-1} A \implies \boxed{H \mathbf{1}_G = \mathbf{1}_G \quad \forall \; \mathcal{R} \text{ covering } G.}$$

### 2.3 Measure-Space Nonnegative Projected SIRT (RMR-v2 / B5-P)
$$Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_c^{-1} A^\top D_a^{-1} (A Y^{(t)} - b) \right] = \max\left(0, \; Y^{(t)} - \omega \cdot r^{(t)}\right).$$
Fixed relaxation $\omega = 1.0$, identity preconditioner $M = 1.0$, detached regional evidence $b$ (`detach_region_evidence: true`), and exact parameter parity with B2 (101,714 params).

### 2.4 Reliability-Weighted Reconciliation (RMR-v3 / RW-RMR)
RMR-v3 extends the transfer operator with per-region uncertainty calibration derived from the Negative-Binomial rate variance:
$$V_R^{\text{rate}} = \frac{\mu_R + \mu_R^2 / r_R}{|R|^2} + \sigma_{\min}^2, \quad w_R = \operatorname{clamp}\left(\frac{\bar{q}_s}{\sqrt{V_R^{\text{rate}}}}, \; 0.25, \; 4.0\right),$$
$$Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_{c,w}^{-1} A^\top W D_a^{-1} (A Y^{(t)} - \mu) \right].$$
Total trainable parameters: **101,763** (< 105,000 budget).

---

## 3. Registered Stage C Matrix (B0–B5)

| ID | Variant Name | Parameter Count | Regional Head | Measure Space Solver | Operator / Allocator | Causal Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 97,681 | ✗ | ✗ | None | Baseline observer without regional head or solver |
| **B1** | Region Loss | 97,681 | ✗ | ✗ | None | Auxiliary regional rate loss on $AY$ without dual head |
| **B2** | Region Aux | 101,714 | ✓ | ✗ | None | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 100,642 | ✗ | Local Conv ($T=2$) | Local 3x3 DWConv | Local neural refinement without regional constraints |
| **B3b** | Learned Projector | 104,851 | ✓ | Measure ($\Pi_+$, $T=2$) | Learned $P_\theta$ | Learned neural allocator vs exact adjoint $A^\top$ |
| **B5-P**| **RMR-P (Stage C Reference)**| **101,714** | ✓ | Measure ($\Pi_+$, $T=2$) | Exact $D_c^{-1} A^\top D_a^{-1}$ | Exact mathematical adjoint reconciliation (parity with B2) |
| **V3-A**| **Probabilistic Uniform** | **101,763** | ✓ (NB Mean+Disp) | Measure ($\Pi_+$, $T=2$) | Uniform $W=I$ | Probabilistic head under uniform reconciliation |
| **V3-B**| **RW-RMR (Active)** | **101,763** | ✓ (NB Mean+Disp) | Measure ($\Pi_+$, $T=2$) | Weighted $A^\top W$ | Reliability-weighted reconciliation |

---

## 4. Quickstart Guide

### 4.1 Environment Setup
```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 4.2 Running Tests
```powershell
pytest tests/core/ tests/rmr_v2/ tests/rmr_v3/ -v --tb=short
```

### 4.3 Training
**Train RMR-v2 (Stage C)**:
```powershell
python -m rmr_v2.train --config configs/rmr_v2/rmr_projected_t2.yaml --lr 0.0001 --output-dir runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42
```

**Train RMR-v3 (RW-RMR)**:
```powershell
python -m rmr_v3.train --config configs/rmr_v3/reliability_weighted.yaml --lr 0.0001 --output-dir runs/sha_a/rmr_v3_reliability_weighted_seed42
```

### 4.4 Evaluating Checkpoints
```powershell
# RMR-v2
python -m rmr_v2.eval --checkpoint runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42/best_val_mae.pt --manifest data/sha_a_val.jsonl

# RMR-v3
python -m rmr_v3.eval --checkpoint runs/sha_a/rmr_v3_reliability_weighted_seed42/best_val_mae.pt --manifest data/sha_a_val.jsonl
```

### 4.5 Test Set Release Gate
The final test set (`data/sha_a_test.jsonl`) is strictly protected to prevent data leakage and multiple-hypothesis testing bias. To evaluate frozen models after the permanent commit freeze:
```powershell
$env:RMR_ALLOW_TEST_EVAL = "1"
powershell -ExecutionPolicy Bypass -File .\scripts\release\run_final_test_eval.ps1
```

---

## 5. Canonical Documentation

Detailed specifications in `docs/rmr/`:
- [**Paper Specification (CVPR 2026)**](docs/rmr/PAPER_SPEC.md): Derivations, transfer theorems, measure-space SIRT, and causal control claims.
- [**Implementation Specification**](docs/rmr/IMPLEMENTATION_SPEC.md): Dynamic MobileNetV4 reduction probing, FP32 AMP operators, and loss dispatch.
- [**Evaluation Specification**](docs/rmr/EVALUATION_SPEC.md): Canonical NAE, physical GAME, diagnostic traces, and paired significance tests.
