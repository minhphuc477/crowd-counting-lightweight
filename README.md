# RMR-Count: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

[![Tests](https://img.shields.io/badge/pytest-131%2F131%20passed-brightgreen.svg)]()
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/carrier-101.7k%20params-orange.svg)]()
[![Target](https://img.shields.io/badge/venue-CVPR%202026-purple.svg)]()

> **Core Research Question:** In ultra-lightweight crowd counting (< 105k parameters), maintaining both fine spatial cell fidelity and long-range spatial consistency is challenging under strict mobile computation budgets. Because learning dense global self-attention or deep multi-scale dilated receptive fields is parameter-prohibitive, we explore:  
> $$\boxed{\textbf{Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?}}$$  
>  
> **Scope:** RMR-Count is an explicit counting and spatial density estimation model. Point localization, detection bounding boxes, and Hungarian matching are outside the primary causal contribution.

---

## 1. Key Mathematical Contribution: The Adjoint Transfer Operator

Let $Y \in \mathbb{R}_+^G$ be the discrete cell count measure on spatial lattice $G$ (stride $s=4$). The canonical ground truth per discrete cell is:
$$Y_{ij}^* = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left(\left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j\right),$$
with cell coordinates clamped to valid feature grid bounds $[0, H_o - 1] \times [0, W_o - 1]$.

Let $\mathcal{R} = \{R_m\}_{m=1}^M$ be a multi-scale regional dictionary across scales $K \in \{32, 64, 128\}$ px with $50\%$ stride overlap.

### 1.1 Discrete Regional Operators
- **Forward Regional Projection Matrix** $A \in \{0, 1\}^{M \times G}$:
  $$(AY)_m = \sum_{g \in R_m} Y_g = q_m.$$
- **Adjoint Back-Projection Matrix** $A^\top \in \{0, 1\}^{G \times M}$:
  $$(A^\top r)_g = \sum_{m: g \in R_m} r_m.$$

### 1.2 The Adjoint Transfer Theorem ($H \mathbf{1} = \mathbf{1}$)
Let $D_a = \operatorname{diag}(A \mathbf{1}_G) \in \mathbb{R}^{M \times M}$ be regional areas, and $D_c = \operatorname{diag}(A^\top \mathbf{1}_M) \in \mathbb{R}^{G \times G}$ be cell coverage counts.  
The normalized regional transfer operator is defined as:
$$H = D_c^{-1} A^\top D_a^{-1} A.$$

$$\boxed{H \mathbf{1}_G = \mathbf{1}_G \quad \forall \; \mathcal{R} \text{ covering } G.}$$

**Theoretical Significance:** The normalized transfer operator preserves constant fields: a uniform regional rate discrepancy induces an identically uniform spatial correction across all covered cells, regardless of the multi-scale overlap dictionary.

---

## 2. Canonical RMR-v2 Architecture

The architecture uses a pretrained MobileNetV4 carrier, decoupling fine local density prediction from multi-scale regional evidence, followed by unrolled Nonnegative Projected SIRT in measure space:

```
Input Image [1, 3, H, W]
       │
       ▼
Pretrained MobileNetV4-Conv-Small-0.5 (truncated at reduction 16, 87,568 params)
   C4 (stride 4, 16 ch), C8 (stride 8, 32 ch), C16 (stride 16, 48 ch)
       │
       ▼
Additive FPN Neck (~6.9k params, width=32 ch)
   ├── 1x1 lateral projections of C4, C8, C16 -> 32 ch
   ├── P16 context dilation: [1, 2, 3] depthwise dilated blocks
   └── Top-down additive pyramid -> (P4, P8, P16)
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Measure Head (~3.2k params)               Scale-Matched Regional Evidence Head (~4.0k params)
Depthwise-sep Conv3x3 + Conv1x1                Shared P4-coordinate regional support:
init bias: b_0 ≈ -4.142 (softplus ≈ 0.0158)    32px -> P4, 64px -> P8 (up to P4), 128px -> P16 (up to P4)
       │                                       MLP([u_R, log(s_R/32.0)]) -> rate * |R| = b_R
       ▼                                               │
Observer Measure Y_0                                   ▼
       │                                     Regional Evidence b (detached)
       └───────────────────────┬───────────────────────┘
                               ▼
            Nonnegative Projected SIRT (RMR-P, T=2, 0 extra params)
                 │
                 ├── Regional count: q = A Y^(t)
                 ├── Rate residual: r_rate = (q - b) / D_a
                 ├── Adjoint field: r_field = D_c^(-1) A^T r_rate
                 └── Measure projection: Y^(t+1) = max(0, Y^(t) - omega * r_field)
                               │
                               ▼
                     Final Measure Field Y_T
```

### 2.1 Measure-Space Nonnegative Projected SIRT (RMR-P)
RMR-P operates directly in measure space with fixed relaxation $\omega = 1.0$, identity spatial gate $M = 1.0$, and non-negativity projection $\Pi_+$:
$$\boxed{Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_c^{-1} A^\top D_a^{-1} (A Y^{(t)} - b) \right] = \max\left(0, \; Y^{(t)} - \omega \cdot r^{(t)}\right).}$$
Regional evidence $b$ is detached during reconciliation (`detach_region_evidence: true`), causally isolating observer estimation from runtime reconciliation. RMR-P requires **zero additional solver parameters**, achieving exact parameter parity with B2 (Region Aux).

### 2.2 Symmetrical Matched Control: Learned Projector (B3b)
To strictly isolate the mathematical adjoint $D_c^{-1} A^\top D_a^{-1}$ against a learned allocator, B3b is evaluated under identical measure-space dynamics:
$$\boxed{Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot P_\theta(F, Y^{(t)}, A Y^{(t)} - b) \right]}$$
with same $T=2, \omega=1.0$, same detached $b$, and no latent Softplus bottleneck.

---

## 3. Training Objective

The joint multi-scale objective cleanly separates observer mass allocation from magnitude and regional supervision:
$$\boxed{\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{count}}^{\text{NB}_{50}}(Y_T) + 1.0 \cdot \mathcal{L}_{\text{FlatDM16}}^{\kappa=20}(Y_0) + 0.25 \cdot \mathcal{L}_{\text{cell}}(Y_T) + 0.20 \cdot \mathcal{L}_{\text{region}}(b)}$$

1. **$\mathcal{L}_{\text{count}}$ (Negative Binomial, $r=50$)**: Robust count magnitude supervision on final iterate $Y_T$.
2. **$\mathcal{L}_{\text{FlatDM16}}$ (Flat Dirichlet-Multinomial-16, $\kappa=20$)**: Supervises observer spatial mass allocation directly on $Y_0$.
3. **$\mathcal{L}_{\text{cell}}$ (Smooth L1)**: Fine local density calibration on $Y_T$.
4. **$\mathcal{L}_{\text{region}}$ (Smooth L1)**: Supervises regional evidence head predictions $b$.

Optimization: AdamW with differential learning rates (backbone @ $10^{-5}$, neck & task heads @ $10^{-4}$), gradient clipping at 500.0.

---

## 4. Registered Experimental Matrix (B0–B5)

Parameter counts measured via `count_parameters(model)`:

| ID | Variant Name | Parameter Count | Regional Head | Measure Space Solver | Operator / Allocator | Causal Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 97,681 | ✗ | ✗ | None | Baseline observer without regional head or solver |
| **B1** | Region Loss | 97,681 | ✗ | ✗ | None | Auxiliary regional rate loss on $AY$ without dual head |
| **B2** | Region Aux | 101,714 | ✓ | ✗ | None | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 100,642 | ✗ | Local Conv ($T=2$) | Local 3x3 DWConv | Local neural refinement without regional constraints |
| **B3b** | Learned Projector | 104,851 | ✓ | Measure ($\Pi_+$, $T=2$) | Learned $P_\theta$ | Learned neural allocator vs exact adjoint $A^\top$ |
| **B5-P**| **RMR-P (Registered)**| **101,714** | ✓ | Measure ($\Pi_+$, $T=2$) | Exact $D_c^{-1} A^\top D_a^{-1}$ | Exact mathematical adjoint reconciliation (parity with B2) |

---

## 5. Quickstart Guide

### 5.1 Environment Setup
```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 5.2 Dataset Manifests
Generate portable JSONL manifests with boundary coordinate preservation:
```powershell
python -m rmr_count.prepare_manifest `
    --images data/part_A_final/train_data/images `
    --annotations data/part_A_final/train_data/ground_truth `
    --dataset sha_a `
    --out data/sha_a_train.jsonl `
    --relative-to .
```

### 5.3 Stage C Training Matrix (Val-Only)
Train models for 1000 epochs with validation evaluation:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_stage_c_matrix.ps1
```

### 5.4 Final Test Set Benchmark (Post-Freeze)
Evaluate frozen checkpoints on the test set exactly once:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_final_test_eval.ps1
```

---

## 6. Canonical Documentation

Detailed specifications in `docs/rmr/`:
- [**Paper Specification (CVPR 2026)**](docs/rmr/PAPER_SPEC.md): Derivations, transfer theorems, measure-space SIRT, and causal control claims.
- [**Implementation Specification**](docs/rmr/IMPLEMENTATION_SPEC.md): Dynamic MobileNetV4 reduction probing, FP32 AMP operators, and loss dispatch.
- [**Evaluation Specification**](docs/rmr/EVALUATION_SPEC.md): Canonical NAE, physical GAME, diagnostic traces, and paired significance tests.

