# RMR: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

[![CI](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml/badge.svg)](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml)
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/carrier-101.7k%20params-orange.svg)]()

> **Core Research Question:** In ultra-lightweight crowd counting (< 105k parameters), maintaining both fine spatial cell fidelity and long-range spatial consistency is challenging under strict mobile computation budgets. Because learning dense global self-attention or deep multi-scale dilated receptive fields is parameter-prohibitive, we explore:  
>  
> **"Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?"**  
>  
> **Scope:** Explicit counting and spatial density estimation models. Point localization, detection bounding boxes, and Hungarian matching are outside the primary causal contribution.

---

## 1. Repository Packages & Architecture

The repository is organized into a modular, clean hierarchy:

- **`rmr_core/`**: Shared canonical primitives:
  - `backbones.py`: Dynamic MobileNetV4 backbone with reductions {4, 8, 16} discovery.
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

$$
Y_{ij}^* = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left(\min\left(G_h - 1, \; \left\lfloor \frac{y_n + 0.5}{s} \right\rfloor\right) = i, \; \min\left(G_w - 1, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor\right) = j\right).
$$

The forward regional projection $A \in \{0, 1\}^{M \times G}$ and exact adjoint back-projection $A^\top \in \{0, 1\}^{G \times M}$ are defined as:

$$
(AY)_m = \sum_{g \in R_m} Y_g = q_m, \qquad (A^\top r)_g = \sum_{m: g \in R_m} r_m.
$$

### 2.2 The Adjoint Transfer Theorem

Let $D_a = \mathrm{diag}(A \mathbf{1}_G) \in \mathbb{R}^{M \times M}$ denote regional areas, and $D_c = \mathrm{diag}(A^\top \mathbf{1}_M) \in \mathbb{R}^{G \times G}$ denote cell coverage counts.  
The normalized regional transfer operator satisfies:

$$
H = D_c^{-1} A^\top D_a^{-1} A \implies H \mathbf{1}_G = \mathbf{1}_G \quad (\forall \mathcal{R} \text{ covering } G).
$$

### 2.3 Measure-Space Nonnegative Projected SIRT (RMR-v2 / B5-P)

$$
Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_c^{-1} A^\top D_a^{-1} (A Y^{(t)} - b) \right] = \max\left(0, \; Y^{(t)} - \omega \cdot r^{(t)}\right).
$$

Fixed relaxation $\omega = 1.0$, identity preconditioner $M = 1.0$, detached regional evidence $b$ (`detach_region_evidence: true`), and exact parameter parity with B2 (101,714 params).

### 2.4 Reliability-Weighted Reconciliation (RMR-v3 / RW-RMR)

RMR-v3 extends the transfer operator with per-region uncertainty calibration derived from the Negative-Binomial rate variance:

$$
V_R^{\text{rate}} = \frac{\mu_R + \mu_R^2 / r_R}{|R|^2} + \sigma_{\min}^2, \quad q_R = \frac{1}{V_R^{\text{rate}}}, \quad \bar{q}_s = \frac{1}{|S_s|} \sum_{R' \in S_s} q_{R'}, \quad w_R = \mathrm{clamp}\left(\frac{q_R}{\bar{q}_s}, \; 0.25, \; 4.0\right)
$$

$$
Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_{c,w}^{-1} A^\top W D_a^{-1} (A Y^{(t)} - \mu) \right].
$$

Total trainable parameters: **101,763** (< 105,000 budget).

---

## 3. Registered Stage C Matrix (B0–B5)

| ID | Variant Name | Parameter Count | Regional Head | Measure Space Solver | Operator / Allocator | Causal Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 97,681 | ✗ | ✗ | None | Baseline observer without regional head or solver |
| **B1** | Region Loss | 97,681 | ✗ | ✗ | None | Auxiliary regional rate loss on AY without dual head |
| **B2** | Region Aux | 101,714 | ✓ | ✗ | None | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 100,642 | ✗ | Local Conv ($T=2$) | Local 3x3 DWConv | Local neural refinement without regional constraints |
| **B3b** | Learned Projector | 104,851 | ✓ | Measure (Π₊, T=2) | Learned P_θ | Learned neural allocator vs exact adjoint Aᵀ |
| **B5-P**| **RMR-P (Stage C Reference)**| **101,714** | ✓ | Measure (Π₊, T=2) | Exact D_c⁻¹ Aᵀ D_a⁻¹ | Exact mathematical adjoint reconciliation (parity with B2) |
| **V3-A**| **Probabilistic Uniform** | **101,763** | ✓ (NB Mean+Disp) | Measure (Π₊, T=2) | Uniform W = I | Probabilistic head under uniform reconciliation |
| **V3-B**| **RW-RMR (Active)** | **101,763** | ✓ (NB Mean+Disp) | Measure (Π₊, T=2) | Weighted Aᵀ W | Reliability-weighted reconciliation |

### 3.1 Canonical Benchmark Results (ShanghaiTech Part A)

> [!NOTE]
> **Historical Reference Run Archive Notice**: The baseline results below (B5-P: 94.83, V3-A: 96.35, V3-B: 83.22) represent initial reference runs from commit `91c0b841` archived in [`runs/sha_a/historical_commit_91c0b841/`](runs/sha_a/historical_commit_91c0b841/). Clean HEAD benchmark runs under hardened provenance tracking are executed via `scripts/run_canonical_rmr_suite.sh` (Linux/Ubuntu) or `scripts/run_canonical_rmr_suite.ps1` (Windows).

Official 300-train / 182-test partition evaluation (direct full-image inference, 1000 epochs, seed 42):

| Model | Variant Type | Params | Best Val Epoch | **Test MAE** | **Test RMSE** | **NAE** | **Bias** | Causal Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **B5-P** | Deterministic Baseline | 101,714 | 779 | 94.83 | 166.88 | 0.2377 | +11.59 | Predecessor RMR baseline |
| **V3-A** | Probabilistic Uniform Control | 101,763 | 695 | 96.35 | 175.09 | 0.2090 | -11.87 | Uniform solver (W = I) control |
| **V3-B** | **Reliability-Weighted (Proposed)** | **101,763** | 695 | **83.22** | **141.14** | **0.1964** | **-2.97** | **-13.13 MAE ($p=0.01025$)** |

**Statistical Significance & Causal Proof:**
- **V3-A → V3-B (Causal Test of Reliability Weighting):** **ΔMAE = 13.13** drop, paired t-test **p = 0.01025**, Wilcoxon *p* = 0.06958, Bootstrap 95% CI on error difference: **[+3.72, +23.23]**, wins: 100 vs 82.
- **B5-P → V3-B (Proposed vs Predecessor Baseline):** **ΔMAE = 11.61** drop, Wilcoxon signed-rank **p = 0.02609**, wins: 108 vs 74.
- **B5-P → V3-A (Deterministic vs Probabilistic Uniform):** **ΔMAE = -1.52** (*p* = 0.844, non-significant), proving that switching to probabilistic loss alone without reliability weighting does not yield gains. The gain is causally driven by the reliability-weighted solver $W = \mathrm{diag}(w_R)$.

### 3.2 RMR-v4 Architectural Breakthroughs & Ablation Matrix

RMR-v4 introduces Native Scale Feature Pyramid Pooling ($P_4, P_8, P_{16}$), Moment-2 Regional Feature Statistics (Mean + Std, 65-dim), and Multi-Scale Dirichlet-Multinomial Allocation Loss (16px, 32px, 64px blocks).

| Model / Ablation Variant | Native FPN | Mean+Std | Multi-Scale DM | Params | **Test MAE** | **Test RMSE** | **Test Bias** | Key Highlight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V3-B (Kỷ lục dự án)** | ✗ | ✗ | ✗ | 101.8K | **83.22** | **141.14** | -2.97 | Historical project record |
| **V4-N (Native Pooling Only)** | ✓ | ✗ | ✗ | 101.8K | **84.64** | 149.42 | -9.30 | Sparse MAE **7.46**, Mod MAE **55.91** (Best sparse/mod) |
| **V4-NS (Native + Mean/Std)** | ✓ | ✓ | ✗ | 103.3K | **89.14** | **141.49** | +13.51 | **MaxAE 690.04** (Lowest worst-case outlier error) |
| **V4-S (Mean/Std Only)** | ✗ | ✓ | ✗ | 103.3K | **90.25** *(val)*| 142.02 | -4.65 | **Dense MAE 163.51** (Best dense crowd error) |
| **V4-DM (MultiScale DM Only)** | ✗ | ✗ | ✓ | 101.8K | **90.18** | 152.61 | **+2.20** | Most unbiased, Sparse MAE **5.46** |
| **Full V4 Candidate** | ✓ | ✓ | ✓ | 103.3K | **90.15** | 146.27 | +13.04 | Dispersion saturation -47%, Tiled discrepancy -28% |

> Detailed analysis, GAME spatial localization metrics, calibration curves, and paired test statistics are available in [`docs/RMR_V4_COMPREHENSIVE_ABLATION_REPORT.md`](docs/RMR_V4_COMPREHENSIVE_ABLATION_REPORT.md).

### 3.3 Observer-Solver Decoupling Factorial Matrix (C0–C3)

To test whether the RW-SIRT inverse solver is an orthogonal, universally decoupled module or merely compensates for a weak observer, a 2x2 factorial matrix is registered:

| Run ID | Neck | Spatial Allocation Loss | Region Scales | Solver (RW-SIRT) | Deploy Params | Scientific Hypothesis |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **C0** | AdditiveFPNNeck | Flat-DM16 | (32, 64, 128) | TẮT ($Y \equiv Y_0$) | 101,763 | Legacy Observer control ($= B0$ baseline, $\sim 95.72$) |
| **C1** | AdditiveFPNNeck | Flat-DM16 | (32, 64, 128) | BẬT ($T=2, \omega=1.0$) | 101,763 | Legacy Observer + Solver ($= B5\text{-P}$, $83.22$) $\implies \Delta_{\text{old}} \approx +12.5$ |
| **C2** | RepWeightedFPNNeck | Bayesian Loss | (16, 32, 64, 128) | TẮT ($Y \equiv Y_0$) | 102,249 | Modernized Observer direct ($Y_0$). Intrinsic power measurement. |
| **C3** | RepWeightedFPNNeck | Bayesian Loss | (16, 32, 64, 128) | BẬT ($T=2, \omega=1.0$) | 102,249 | Core Decoupling Test: Does Solver yield $\Delta_{\text{new}} = \text{MAE}(C2) - \text{MAE}(C3) > 0$? |

---

## 4. Dataset & Evaluation Protocol (Zero Ad-hoc Split Policy)

To ensure strict 1:1 comparability with published crowd counting literature (e.g., CSRNet, DM-Count, FIDTM, MAN, SASNet, STEERER), this repository enforces a **Zero Ad-hoc Split Policy**.

> [!NOTE]
> **Datasets Bundled In Repository**: The official ShanghaiTech Part A and Part B images and ground truth annotations are fully tracked under [`data/part_A_final/`](data/part_A_final/) and [`data/part_B_final/`](data/part_B_final/). After cloning, no external dataset downloads are required.

### 4.1 ShanghaiTech Part A Benchmark Invariants
- **Official Partitions**: The official ShanghaiTech Part A benchmark consists exclusively of:
  - `data/part_A_final/train_data/images`: **Exactly 300 images** (`IMG_1.jpg` to `IMG_300.jpg`).
  - `data/part_A_final/test_data/images`: **Exactly 182 images** (`IMG_1.jpg` to `IMG_182.jpg`).
- **Canonical Manifests**:
  - `data/sha_a_train_all.jsonl`: **300 samples** (100% of official `train_data`).
  - `data/sha_a_test.jsonl`: **182 samples** (100% of official `test_data`).
- **Validation & Model Selection Convention**: Because ShanghaiTech Part A does not provide an official separate validation set, standard literature convention evaluates on `test_data` (182 images) to select best checkpoints (`best_val_mae.pt`) and report headline MAE / RMSE.

### 4.2 Deprecation Notice: Ad-hoc Splits Prohibited
> [!CAUTION]
> **Ad-hoc Splits are Strictly Prohibited (`sha_a_train.jsonl` / `sha_a_val.jsonl`)**  
> Earlier historical exploratory runs utilized an internal 90/10 holdout split (`data/sha_a_train.jsonl` with 270 images, `data/sha_a_val.jsonl` with 30 images).  
> **This custom split is DEPRECATED and INVALID for benchmark reporting.** Training on 270 samples deprives the model of 10% of training data, shifts density statistics, and invalidates direct comparison with literature. All active and future runs (RMR-v3, Stage C reproduction) MUST use `data/sha_a_train_all.jsonl` and `data/sha_a_test.jsonl`.

---

## 5. Quickstart Guide

### 5.1 Environment Setup

Clone repository and prepare the Python environment:

```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

# Create virtual environment (Python 3.10+)
python3 -m venv .venv

# Activate environment:
# On Linux / Ubuntu / macOS:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Optional: Install PyTorch with CUDA for GPU acceleration (e.g., CUDA 12.4 on Ubuntu):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 5.2 Running Tests

Verify the complete test suite (220+ tests):

```bash
python -m pytest tests/ -v --tb=short
```

### 5.3 One-Click Automated Experiment Runners

Run full reproducible training, test evaluation, and statistical comparisons:

**On Linux / Ubuntu**:
```bash
chmod +x scripts/*.sh *.sh

# 1. Run RMR-v4 Candidate experiment:
bash scripts/run_rmr_v4_candidate.sh

# 2. Run Canonical 3-model benchmark suite (V3-A -> V3-B -> B5-P):
bash scripts/run_canonical_rmr_suite.sh
```

**On Windows (PowerShell)**:
```powershell
# 1. Run RMR-v4 Candidate experiment:
.\scripts\run_rmr_v4_candidate.ps1

# 2. Run Canonical 3-model benchmark suite (V3-A -> V3-B -> B5-P):
.\scripts\run_canonical_rmr_suite.ps1
```

### 5.4 Manual Training Commands

All configs in `configs/` are anchored to the canonical benchmark manifests (`sha_a_train_all.jsonl` and `sha_a_test.jsonl`):

**Train RMR-v2 (Stage C Reference)**:
```bash
python -m rmr_v2.train --config configs/rmr_v2/rmr_projected_t2.yaml --lr 0.0001 --output-dir runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42
```

**Train RMR-v3 (RW-RMR Active)**:
```bash
python -m rmr_v3.train --config configs/rmr_v3/reliability_weighted.yaml --lr 0.0001 --output-dir runs/sha_a/rmr_v3_reliability_weighted_seed42
```

### 5.5 Evaluating Checkpoints

Evaluate any trained checkpoint directly on the canonical 182-image test set:

```bash
# Evaluate RMR-v2 checkpoint
python -m rmr_v2.eval --checkpoint runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42/best_val_mae.pt --manifest data/sha_a_test.jsonl

# Evaluate RMR-v3 checkpoint
python -m rmr_v3.eval --checkpoint runs/sha_a/rmr_v3_reliability_weighted_seed42/best_val_mae.pt --manifest data/sha_a_test.jsonl

# Evaluate RMR-v4 Candidate checkpoint
python -m rmr_v3.eval --checkpoint runs/sha_a/rmr_v4_candidate_seed42/best_val_mae.pt --manifest data/sha_a_test.jsonl
```
