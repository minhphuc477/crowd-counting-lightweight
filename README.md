# RMR: Lightweight Radon Measure Recovery for Crowd Counting

[![CI](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml/badge.svg)](https://github.com/minhphuc477/crowd-counting-lightweight/actions/workflows/ci.yml)
[![Branch](https://img.shields.io/badge/branch-RMR-blue.svg)]()
[![Parameters](https://img.shields.io/badge/parameters-Track%20A%20%E2%89%A4104.4k%20%7C%20Track%20B%20%E2%89%A4200k-orange.svg)]()
[![Knowledge Distillation](https://img.shields.io/badge/knowledge%20distillation-0.0%25%20(standalone)-green.svg)]()
[![Champion Benchmark](https://img.shields.io/badge/champion%20MAE-69.03%20(V64)%20%7C%2070.02%20(V72)-brightgreen.svg)]()

> **Core Research Question:** In extreme ultra-lightweight crowd counting ($\le 104,441$ parameters for Stride 4; $\le 200,000$ parameters for Subpixel Stride 2, with exactly $0.0\%$ Knowledge Distillation), how can ill-posed discrete point counting be mathematically formulated as an inverse measure recovery problem without suffering from spatial aliasing, scale collapse, or phantom background fog?
>
> **Mathematical Formulation:**
> $$\text{Fredholm Integral Operator: } b = \mathcal{A}_\pi y + \eta$$
> $$\text{Closed-Form Bregman Projection: } y^* = y_0 \cdot \mathcal{A}_\pi^T \left( \frac{b}{\mathcal{A}_\pi y_0 + \epsilon} \right)$$

---

## 1. Project Invariants & Operational Standards

All architectures and experiments strictly adhere to [`RULES.md`](RULES.md) and [`GEMINI.md`](GEMINI.md):
1. **Strict Parameter Budgets**:
   - **Track A (Stride 4 Anchor)**: Strictly $\le 104,441$ trainable parameters.
   - **Track B (Subpixel Stride 2 Capacity Extension)**: Strictly $\le 200,000$ trainable parameters.
2. **100% Standalone (Zero Knowledge Distillation)**:
   - Exactly $0.0\%$ Knowledge Distillation. Zero teacher networks at training or inference.
3. **Empirical Integrity**:
   - Zero fabricated metrics, zero oracle substitutions in reported benchmarks.
   - Full evaluation on all 182 official ShanghaiTech Part A test images via standard MAE / RMSE / GAME(0,1,2,3) / NAE protocols.
4. **Code Quality & Maintainability**:
   - Strictly $\le 450$ lines per Python source file.
   - Pure Python / YAML configurations.
5. **Zero Frankencode & Anti-Patchwork Policy**:
   - Strictly forbidden to arbitrarily cobble together heuristics or ad-hoc patches. Every component must have rigorous mathematical/physical justification.

---

## 2. Repository Architecture & Layout

```text
lightweightcrcn/
├── configs/
│   ├── rmr_v64/            # Champion baseline configurations (MAE 69.03)
│   ├── rmr_v72/            # Active Track B Stride 2 anchor suite (MAE 70.02)
│   ├── rmr_research/       # Core historical research configs (regression tested)
│   └── rmr_v*/             # Historical iteration configs (v2 to v70)
├── data/
│   ├── part_A_final/       # ShanghaiTech Part A official benchmark images and GT
│   ├── part_B_final/       # ShanghaiTech Part B official benchmark images and GT
│   ├── sha_a_train_all.jsonl # 300 official training samples
│   └── sha_a_test.jsonl    # 182 official test samples
├── docs/                   # Authoritative technical documentation and audits
│   ├── archive/            # Historical design documents and legacy reports
│   ├── code_explanation/  # 6 modular walkthroughs of architecture, solver, loss, and training
│   ├── CANONICAL_DATASET_CONFIG_RULES_AND_SPECIFICATIONS.md
│   ├── CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md
│   ├── ML_MODEL_DEEP_AUDIT_AND_CODE_REVIEW_PROTOCOL.md
│   └── TOAN_BO_THANH_PHAN_KIEN_TRUC_MODEL_RMR.md
├── rmr_core/               # Shared canonical primitives
│   ├── backbones.py        # MobileNetV4 truncated encoder (reductions 4, 8, 16)
│   ├── necks/              # RepASPPLite, ASPPLite, RLKHDC, Additive, BiFPN
│   ├── heads.py            # Fine carrier density heads with analytical priors
│   ├── operators/          # Box integrals, prefix sums, adjoint transfer, MC-BPRO
│   ├── dm_loss.py          # Flat & Multi-scale Dirichlet-Multinomial allocation loss
│   ├── count_loss.py       # Macro Negative-Binomial count loss
│   └── metrics.py          # NAE, GAME(L) with fractional cell boundary integration
├── rmr_v3/                 # Active research engine
│   ├── config/             # Strict YAML schema validation & architecture contracts
│   ├── losses/             # Orthogonal loss orchestration (DM16, Bayesian, Radon-OT)
│   ├── model/              # Unified RMR architecture, dual-lattice & solver step
│   ├── optim/              # Cosine Annealing, SafeProdigy, Schedule-Free factories
│   ├── regional_head.py    # Probabilistic Regional Evidence Head (Beta-NB)
│   ├── solver.py           # Unrolled iterative SIRT solver (baseline)
│   ├── trainer.py          # Core training orchestrator
│   ├── train.py            # Training entrypoint wrapper
│   └── eval.py             # Evaluation module (full 182-image benchmark)
├── tests/                  # Over 1,600 automated unit & regression tests
├── tools/                  # Analytical CLI tools (profiling, ONNX, error audits)
├── CHANGELOG.md            # Comprehensive project release notes & milestones
├── DERIVATION_PACKAGE.md   # Mathematical derivations from first principles
├── GEMINI.md               # Operational research invariants & server configuration
└── RULES.md                # Strict operational and engineering rules
```

---

## 3. Official Execution Entrypoints

All training and evaluation workflows are executed via standard Python modules:

### 3.1 Training Entrypoint
```bash
# Train champion model (automatic output directory creation):
python train.py --config configs/rmr_v72/v72_track_b_exp0_v64_anchor_stride2.yaml

# Run on specific GPU in background on Linux workstation:
CUDA_VISIBLE_DEVICES=0 nohup python train.py \
  --config configs/rmr_v72/v72_track_b_exp0_v64_anchor_stride2.yaml \
  > train_v72_exp0.log 2>&1 &
```

### 3.2 Full Test Evaluation Entrypoint
```bash
# Evaluate checkpoint on all 182 test images of ShanghaiTech Part A:
python -m rmr_v3.eval --checkpoint runs/sha_a/rmr_v72/v72_track_b_exp0_v64_anchor_stride2/best_val_mae.pt

# Evaluate with Test-Time Augmentation (TTA with L1 mass conservation):
python -m rmr_v3.eval --checkpoint runs/sha_a/rmr_v72/v72_track_b_exp0_v64_anchor_stride2/best_val_mae.pt --tta
```

---

## 4. Official Benchmark Matrix & Verified Experimental Champions

All benchmarks reported below are evaluated strictly on all 182 official test images of ShanghaiTech Part A with zero teacher distillation (100% standalone, 0.0% KD):

| Experiment / Checkpoint Identifier | Track | Params | Stride | Neck | Spatial Allocation Loss | Solver Mode | Val MAE | Val RMSE | Key Significance & Verified Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`v69_exp1_track_a_dual_bayesian_radon_ot_1500ep`** | **Track A** | **104,390** | 4 | RepASPPLite | Dual Bay + Radon-OT | mcbpro | **65.33** | **108.39** | **ALL-TIME BEST TRACK A CHAMPION (TTA)** ($\le 104,441$ params) |
| **`v70_track_b_exp4b_no_solver`** | **Track B** | **122,618** | 2 | RepASPPLite | Flat DM16 | direct (no solver) | **65.69** | **108.08** | **ALL-TIME BEST TRACK B CHAMPION** ($\le 200,000$ params) |
| **`v69_exp1_track_a_dual_bayesian_radon_ot_1500ep`** | **Track A** | 104,390 | 4 | RepASPPLite | Dual Bay + Radon-OT | mcbpro | **65.82** | **110.47** | Track A Standard Val split |
| **`v64_champion_synergy_50mae_1500ep`** | **Track B** | 197,839 | 2 | RepASPPLite | Flat DM16 | mcbpro | **69.03** | **118.24** | Verified V64 Anchor Baseline |
| **`v72_track_b_exp0_v64_anchor_stride2`** | **Track B** | 197,839 | 2 | RepASPPLite | Flat DM16 | mcbpro | **70.02** | **107.86** | Active Track B V72 Reproduction Checkpoint |
| **`v67_exp4_abl_pure_2loss_no_dm16_1500ep`** | **Track B** | 197,839 | 2 | RepASPPLite | Pure 2-Loss (Bay + Count) | mcbpro | **70.85** | **111.97** | Elimination of Flat DM16 conflict |
| **`v65_champion_pure_synergy_1500ep`** | **Track B** | 197,839 | 2 | RepASPPLite | Dual Bay + DM16 | mcbpro | 73.12 | 129.54 | Dual Allocation Ablation |
| **`v63_champion_aspp_subpixel_1500ep`** | **Track B** | 125,786 | 2 | ASPPLite | Flat DM16 | mcbpro | 74.54 | 134.12 | Initial Stride 2 Subpixel Milestone |

---

## 5. First-Principles Breakthroughs & Current Frontier

Through exhaustive microscopic-to-macroscopic forensic auditing across all 182 test images and mathematical analysis of loss gradients, the fundamental bottleneck preventing models from achieving Sub-50 MAE has been proven:
1. **The Dynamic Nash Equilibrium Trap**:
   Under the composite loss $\mathcal{L} = \lambda_{\text{count}} \mathcal{L}_{\text{count}} + \lambda_{\text{alloc}} \mathcal{L}_{\text{alloc}}$, the network reaches an equilibrium where real crowd peaks receive only $\approx 35.6\%$ of their mass, while the remaining $\approx 64.4\%$ is supplied by an imperceptible background fog ($\approx 0.00124\text{ count/cell}$).
2. **Gradient Vanishing of Convex Activations**:
   The Softplus activation $y = \tau \log(1 + e^{z/\tau})$ has an exponentially vanishing derivative $\sigma(z/\tau) \to 0$ for negative background logits $z \approx -7.7$, causing background suppression gradients to vanish by a factor of over $2,000\times$, locking the background bias permanently at $b \approx -5.442$.
3. **Spatial Averaging Blindness of Regional Head**:
   Average pooling across large spatial boxes dilutes micro-head signals by $99.5\%$, reducing the Pearson linear correlation of the regional evidence head with ground truth to $r = 0.0521$.
4. **Resolution Roadmap**:
   Solving this requires reforming the measure support topology from strictly positive Lebesgue-continuous densities to compact support measures, eliminating the background shortcut and forcing 100% of the crowd mass back onto true head peaks.
