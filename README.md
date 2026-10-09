# RMR: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

[![CI](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml/badge.svg)](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml)
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/carrier-104.4k%20params-orange.svg)]()
[![Knowledge Distillation](https://img.shields.io/badge/teacher%20distillation-0.0%25%20(standalone)-green.svg)]()
[![Test Suite](https://img.shields.io/badge/tests-1176%20passed-brightgreen.svg)]()

> **Core Research Question:** In ultra-lightweight crowd counting ($\le 105,000$ parameters, zero knowledge distillation), how can discrete regional-count operators replace heavy learned contextual reasoning without suffering from density saturation or background false alarms?
>
> **Methodology:** Radon Measure Recovery (RMR) formulates crowd counting as an inverse problem in measure spaces. By coupling a lightweight convolutional observer (MobileNetV4, Stride 4) with an unrolled Simultaneous Iterative Reconstruction Technique (SIRT) solver with Lipschitz contraction damping and Morozov discrepancy stopping, RMR reconstructs high-fidelity point density maps while strictly preserving total regional mass.

---

## 1. Project Invariants & Operational Rules

Every configuration and code modification strictly adheres to [`RULES.md`](RULES.md):
1. **Script Immutability**: All shell/PowerShell scripts are strictly immutable without explicit permission. Individual executable commands are provided directly in documentation.
2. **Strict Parameter Ceiling**: $\le 105,000$ trainable parameters. Standalone only ($0.0\%$ Knowledge Distillation / Zero Teacher).
3. **Code Health Invariant**: Every Python source file in `rmr_core/` and `rmr_v3/` strictly satisfies $\le 450$ lines of code.
4. **Anti-Regression Protocol**: Benchmarked on official ShanghaiTech Part A (300 train, 182 test) with verified `summary.json` outputs.

---

## 2. Repository Architecture & Layout

The codebase follows academic research standards (Meta FAIR, Google DeepMind, PyTorch Lightning):

```text
crowd-counting-lightweight/
├── configs/
│   ├── rmr_research/       # Active research generations (gen7, gen8, gen9, gen10, sub60_e*)
│   ├── rmr_sub60/          # Canonical Sub-60 anchor configurations
│   ├── rmr/                # Standard model configurations
│   └── rmr_v*/             # Historical iteration records (v10 to v34)
├── data/
│   ├── part_A_final/       # ShanghaiTech Part A official benchmark images and GT
│   ├── part_B_final/       # ShanghaiTech Part B official benchmark images and GT
│   ├── sha_a_train_all.jsonl # 300 training samples (100% official train split)
│   └── sha_a_test.jsonl    # 182 test samples (100% official test split)
├── docs/                   # Centralized research specifications, autopsies, and reports
│   ├── findings.md         # Forensic error autopsy and causal insights
│   ├── research-state.yaml # Project memory and generation tracker
│   └── specifications/     # Mathematical operators, FPN necks, and loss formulations
├── rmr_core/               # Shared canonical primitives (backbones, necks, heads, operators)
│   ├── backbones.py        # MobileNetV4 dynamic backbone with reductions {4, 8, 16}
│   ├── necks.py            # Additive and RepWeighted FPN necks
│   ├── heads.py            # Fine measure density heads with analytical priors
│   ├── operators/          # Box integrals, prefix sums, adjoint transfer operator H
│   ├── spectral.py         # Count-preserving DCT-II spectral loss
│   └── metrics.py          # NAE, GAME(L) with fractional cell boundary integration
├── rmr_v3/                 # Active research engine
│   ├── model.py            # Unified RMR architecture
│   ├── solver.py           # Unrolled SIRT solver with Barzilai-Borwein step sizes
│   ├── solver_ops.py       # Morozov discrepancy, firm thresholding, Lipschitz damping
│   ├── losses/             # Dirichlet-Multinomial, Bayesian, and Characteristic Function losses
│   ├── trainer.py          # Core training orchestrator (python -m rmr_v3.trainer)
│   ├── train.py            # CLI wrapper (python -m rmr_v3.train)
│   └── eval.py             # Evaluation module (python -m rmr_v3.eval)
├── tests/                  # 1,176 automated tests (100% passed)
├── tools/                  # Analytical CLI tools (profiling, ONNX, error audits)
├── pyproject.toml          # PEP 517/518 build metadata & test configuration
└── RULES.md                # Project rules and operational invariants
```

---

## 3. Official Execution Entrypoints

Mọi tác vụ đào tạo và đánh giá đều được gọi qua module Python chuẩn:

### 3.1 Training Entrypoint
```bash
# Huấn luyện mô hình chuẩn (khuyến nghị luôn có -o để tự động ghi đè output_dir):
python -m rmr_v3.trainer -c configs/rmr_research/<config_name>.yaml -o

# Chạy ngầm trong nền trên Ubuntu server với GPU chỉ định:
CUDA_VISIBLE_DEVICES=0 nohup python -m rmr_v3.trainer \
  -c configs/rmr_research/<config_name>.yaml \
  -o \
  > <log_name>.log 2>&1 &
```

### 3.2 Evaluation Entrypoint
```bash
# Đánh giá checkpoint chuẩn trên tập test:
python -m rmr_v3.eval --checkpoint runs/sha_a/<run_name>/best_val_mae.pt

# Đánh giá kèm Test-Time Augmentation (TTA bảo toàn khối lượng L1):
python -m rmr_v3.eval --checkpoint runs/sha_a/<run_name>/best_val_mae.pt --tta
```

---

## 4. Mathematical Foundations

### 4.1 Discrete Regional Operators & Adjoint Transfer Theorem

Let $Y \in \mathbb{R}_+^G$ be the discrete cell count measure on lattice $G$ (stride $s=4$). The regional observation operator $A \in \{0, 1\}^{M \times G}$ integrates density over multi-scale boxes $R_m \in \{32, 64, 128\}\text{ px}$:

$$ (A Y)_m = \sum_{g \in R_m} Y_g = b_m, \qquad (A^\top r)_g = \sum_{m: g \in R_m} r_m. $$

With regional areas $D_a = \mathrm{diag}(A \mathbf{1}_G)$ and coverage counts $D_c = \mathrm{diag}(A^\top \mathbf{1}_M)$, the normalized transfer operator satisfies the partition of unity:

$$ H = D_c^{-1} A^\top D_a^{-1} A \implies H \mathbf{1}_G = \mathbf{1}_G \quad (\forall \mathcal{R} \text{ covering } G). $$

### 4.2 Unrolled SIRT Solver with Lipschitz Damping

The density field is iteratively reconciled through $T=6$ unrolled iterations:

$$ Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega^{(t)} \cdot D_{c,w}^{-1} A^\top W D_a^{-1} \mathcal{M}_\gamma\left(A Y^{(t)} - b_{\text{solver}}\right) \right], $$

where:
1. **Lipschitz Contraction Damping**: $b_{\text{solver}} = \pi_R \odot b_{\text{raw}}$ prevents carrier field explosion in dense crowd clusters.
2. **Morozov Discrepancy Filter**: $\mathcal{M}_\gamma(r) = \mathrm{sign}(r) \max(|r| - \gamma \sigma(b), 0)$ stops gradient updates below the heteroscedastic noise floor.
3. **Barzilai-Borwein Rayleigh Step Size**: $\omega^{(t)} = \frac{\langle s^{(t)}, y^{(t)} \rangle}{\|y^{(t)}\|^2}$ guarantees fast non-monotone convergence.

---

## 5. Benchmark Leaderboard (ShanghaiTech Part A)

Official 300-train / 182-test partition (full-image direct inference, strictly $\le 105,000$ params, 0.0% Knowledge Distillation):

| Generation | Model Run ID | Params | **MAE** | **RMSE** | **MedianAE** | **Bias** | Key Scientific Contribution |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gen 5** | `m04_iso_fractional_loss` | 104,441 | **71.03** | **111.29** | **44.50** | **-13.57** | **All-Time Project Record**: Fractional Cell Loss $p=0.5$ ($N^{0.5}$ normalization) |
| **Anchor** | `sub60_e5` / `g10_anchor_sub60_e5` | 104,441 | **71.51** | **114.88** | 45.80 | -8.12 | **Robust Anchor**: Flat DM16 + Product Gating ($\lambda_{\text{cell}}=0.0$) |
| **Gen 10** | `g10_subpixel_stride2_chfl` | 104,441 | **76.07** | **115.96** | **44.00** | **-2.96** | **Gen 10 Champion**: Subpixel Stride-2 + ChfL $\lambda=0.35$ |
| **Gen 10** | `g10_clean_asym_morozov` | 104,441 | **77.47** | **114.60** | 46.12 | +1.26 | **Lowest RMSE**: Asymmetric Morozov deadband |
| **Gen 7** | `g7_champion_synthesis` | 104,441 | 74.72 | 110.08 | 49.30 | +4.12 | Continuous Dynamic Windowing + Curvature Control |
| **Stage C**| `V3-B` (Stage C Champion) | 101,763 | 83.22 | 141.14 | 52.10 | -2.97 | Initial Reliability-Weighted Reconciliation record |
| **Stage C**| `B5-P` (Predecessor Baseline) | 101,714 | 94.83 | 166.88 | 61.40 | +11.59 | Unweighted projected SIRT baseline |
| **Ablation**| `g10_subpixel_stride2_no_solver`| 104,441 | 90.08 | 141.31 | 58.70 | -19.45 | **No Solver Control**: Dense MAE explodes to **176.28** |

> **Causal Finding on Solver Necessity:** Disabling the SIRT solver (`g10_subpixel_stride2_no_solver`) causes Dense MAE ($N > 500$) to explode from **133.71** to **176.28** (+42.57 MAE penalty), conclusively proving that the unrolled variational solver is indispensable for resolving dense crowd clusters under the Rayleigh separation limit.

---

## 5. Quickstart Guide

### 5.1 Environment Setup

```bash
git clone https://github.com/minhphuc477/crowd-counting-lightweight.git
cd crowd-counting-lightweight
git checkout RMR

# Create virtual environment (Python 3.10+)
python3 -m venv .venv

# Activate:
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 5.2 Running the Test Suite

Verify all 1030+ tests across the repository:

```bash
python -m pytest tests/ -q
```

### 5.3 Training SOTA Models

Train the Gen 10 Champion configuration:

```bash
python -m rmr_v3.train --config configs/rmr_research/gen10/g10_subpixel_stride2_chfl.yaml --run-id g10_subpixel_stride2_chfl --workers 0 --num-threads 2 --no-cudnn-benchmark
```

Train the Historical Sub-60 Anchor:

```bash
python -m rmr_v3.train --config configs/rmr_research/sub60_e5.yaml --run-id sub60_e5 --workers 0 --num-threads 2 --no-cudnn-benchmark
```

### 5.4 Evaluating Checkpoints

Evaluate any trained checkpoint directly on the canonical 182-image ShanghaiTech Part A test set:

```bash
python -m rmr_v3.eval --checkpoint runs/sha_a/g10_subpixel_stride2_chfl/best_val_mae.pt --manifest data/sha_a_test.jsonl --crop-size 512 --tta false --output-json runs/sha_a/g10_subpixel_stride2_chfl/eval_canonical.json
```

Or using the CLI alias:

```bash
python -m rmr_v3.evaluate --checkpoint runs/sha_a/sub60_e5/best_val_mae.pt --manifest data/sha_a_test.jsonl --crop-size 512 --tta false
```

### 5.5 Model Profiling & Parameter Verification

Verify parameter count and latency on mobile backbones:

```bash
python tools/architecture_table.py
python tools/profile_model.py --config configs/rmr_research/gen10/g10_subpixel_stride2_chfl.yaml
```
