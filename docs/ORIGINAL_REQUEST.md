# Original User Request

## 2026-09-18T15:59:12Z

# Teamwork Project Prompt

Working directory: f:/lightweightcrcn
Integrity mode: benchmark

## 1. Project Overview & Empirical Ground Truth
The project objective is to develop and evaluate the **Regional Measure Reconciliation (RMR)** ultra-lightweight crowd counting framework (under a strict hard ceiling of <= 105,000 trainable parameters, evaluated on canonical ShanghaiTech Part A with 300 train / 182 test images, TTA horizontal flip, zero KD, zero teacher models).

### Key Empirical Milestones & Autopsy:
- **RMR-v19 Canonical**: Achieved **72.61 TTA MAE / 72.84 Direct MAE** with **95% CI [61.13, 85.10]** (Lower bound 61.13), with best-in-class Dense MAE of **122.71** and high stability (80 epochs < 80 MAE, 11 epochs < 75 MAE). Core engine: pure BB-1 Rayleigh contraction step size, clean macro-isotropic geometry [32, 64, 128] px, Radon-Nikodym measure adjoint (support invariance), Bayesian Morozov discrepancy deadband (gamma=0.75), and convex loss core (Mass-weighted cell loss + Flat DM16 + Regional NB).
- **RMR-v25 Reality**: Achieved all-time project record on sparse crowds (**16.38 Sparse MAE** in `rmr_v25_ablation_no_bb`, vs v19's 20.01) via 1D projective camera elevation, but suffered from unnormalized multiplicative scaling P4 * (1 + tanh(W v + b)) that inflated foreground features in the lower half of images, causing net prediction bias to swing by +13.45 into positive over-counting (+4.29 vs v19's -9.16) and interacting with curvature loss to bottleneck overall MAE at 75.55.
- **The Core Research Question (RQ) for the A* Paper**:
  "Can known discrete regional-count operators replace learned contextual reasoning in ultra-lightweight models?"
  Under fixed parameter budget <= 105k, operator-guided unrolled inverse solving with Adjoint Scale Invariance (H 1 = 1) replaces 20M-parameter context networks.

---

## 2. Requirements & Unified Architecture (RMR-v26)

### R1. Mass-Conserved Zero-Inflation Perspective Modulation (MPE-v2)
1. Normalize the 1D elevation modulation field so that its spatial vertical mean across the image height is identically 1.0:
   M(v) = 1.0 + tanh(W v + b)
   M_bar = (1 / H) * sum_{i=1}^H M(v_i)
   P4_tilde(v) = P4(v) * (M(v) / M_bar)
2. Mathematical Invariant: (1 / H) * sum_{i=1}^H (P4_tilde(v_i) / P4(v_i)) == 1.0000. Total carrier mass is strictly conserved.
3. Completely prevents false density inflation in the lower foreground while preserving 1D projective camera depth calibration.
4. Total trainable parameters: exactly 64 parameters (identity warm-start with W=0, b=0).

### R2. Pure BB-1 Rayleigh Contraction with Trust Damping
1. Retain pure Barzilai-Borwein BB-1 step size:
   alpha_1 = <s_k, r_k> / (||r_k||^2 + eps)
2. Tighten the clamping range to [0.5 * omega_0, 1.2 * omega_0] (instead of [0.2, 2.0]) to eliminate aggressive gradient swings in hyper-dense crowd clusters.

### R3. Curvature Loss Damping / Lock to Convex Loss Core
1. Lock auxiliary curvature loss to lambda_curv = 0.0 (or heavily damped lambda_curv <= 0.1).
2. Rely strictly on the proven convex loss pair from v19:
   - Mass-Weighted Cell L1 Loss (lambda_cell = 0.5, alpha = 2.0, gamma = 1.25)
   - Flat-DM16 Optimal Transport Loss (lambda_alloc = 1.0, kappa = 20.0)
   - Scale-Balanced Hurdle-NB Regional Loss (lambda_reg = 0.2)

### R4. Complete 6-Model Verification & Ablation Suite
1. `rmr_v26_canonical`: Mass-Conserved Elevation + Pure BB-1 + Zero Curvature Loss.
2. `rmr_v26_ablation_no_elevation`: Control without elevation modulation (verifying recovery of v19 baseline).
3. `rmr_v26_ablation_no_bb`: Fixed step size omega = 1.0.
4. `rmr_v26_ablation_no_morozov`: Testing noise floor regularization necessity.
5. `rmr_v26_ablation_with_curv01`: Controlled test of mild curvature (lambda = 0.1).
6. `rmr_v26_control_no_solver`: Feedforward baseline (Y_0).

---

## 3. Acceptance Criteria

### Mathematical Rigor
- [ ] Strict mass conservation verified: (1/H) * sum (P4_tilde / P4) == 1.0000 +- 1e-6.
- [ ] Trainable parameters strictly verified <= 105,000 (target: 104,505 params).
- [ ] Radon-Nikodym support absorption verified: zero background phantom count.

### Empirical Validation
- [ ] 100% unit test pass rate across all suite tests.
- [ ] Evaluation on canonical ShanghaiTech Part A (182 test images, TTA, stride 4):
  - Overall MAE <= 70.0 (target Sub-65).
  - Sparse MAE <= 18.0 (retaining v24/v25 breakthrough).
  - Dense MAE <= 125.0 (retaining v19 breakthrough).
  - Net prediction bias within [-5.0, +2.0] (zero inflation).

## 2026-09-20T02:02:51Z

RMR-v30: Breaking the Sub-60 MAE Barrier in Ultra-Lightweight Crowd Counting (<= 105k params) via Comprehensive Mathematical Physics: Dual-Lattice Radon Measure Recovery, Anscombe Variance-Stabilizing SIRT Inversion, and Adaptive Receptive Inversion on Canonical ShanghaiTech Part A.

Working directory: f:/lightweightcrcn
Integrity mode: benchmark

Requested team: Standard full research team (1 orchestrator, 1 math-physics lead, 1 architecture engineer, 1 test/eval auditor)
Research Directive: Nghiên cứu toàn diện — đa nhánh giả thuyết (Dual-Lattice DCSR + Anscombe VST + Deep SIRT T=8 + Adaptive Relaxation)

## 1. Executive Scientific Synthesis (Autoresearch Outer Loop: v19 to v29)

### 1.1 The Forensic Evidence Landscape
Across the rigorous experimental trajectory from v19 to v29 on ShanghaiTech Part A:
- RMR-v19 Anchor (Proven Baseline): 72.84 Direct MAE (72.61 TTA MAE) with 104,441 parameters, Stride 4, Macro-isotropic regions [32, 64, 128], BB step clamp [0.2, 2.0], mass-weighted cell loss + flat DM16 + regional hurdle NB.
- RMR-v29 H2 Empirical Breakthrough: Sparse MAE reached 10.64 (an all-time project record, 2x better than v19's 22.06), proving that higher spatial discretization (Stride 2) resolves isolated and sparse heads with exceptional precision.
- The Stride-2 Dense Failure Mode: In v29 H2, while Sparse MAE was 10.64, Dense MAE exploded to 191.32 (with a net bias of -29.95, representing severe undercounting). 
  - Mathematical Cause: In dense crowds (N > 500), the discrete Dirac impulse distribution on a Stride-2 lattice (256x256) experiences severe gradient starvation: the background cell area quadruples (16x16 = 256 cells per 32px box), causing the Radon-Nikodym adjoint denominator q + eps * area to over-damp updates by 4x, while the unscaled proximal threshold tau truncates the low-magnitude density tails of dense clumps.

### 1.2 The Sub-60 Mathematical Path
Target MAE = [15 * Sparse(10.64) + 120 * Moderate(<= 50.0) + 47 * Dense(<= 102.0)] / 182 <= 59.8
To break the 60 MAE barrier, RMR-v30 must:
1. Preserve the 10.64 Sparse MAE from Stride-2 spatial resolution.
2. Eliminate the -29.95 undercounting bias in dense crowds by coupling Stride-2 fine measure with Stride-4 carrier density and Anscombe Variance-Stabilizing Transformation (VST).
3. Maintain strict mathematical invariants: zero Dirac mass inflation, support-preserving Radon-Nikodym adjoint, and strict <= 105,000 trainable parameters.

## 2. Requirements

### R1. Dual-Lattice Consistency & Density-Conditioned Spatial Resolution (DCSR)
The model must represent density measures across dual spatial lattices (Stride 4 carrier lattice for dense stability, Stride 2 fine lattice for sparse precision) or dynamically modulate the proximal threshold tau(x, y) and Radon-Nikodym regularizer as a function of local density:
tau_eff(x, y) = tau_0 * (stride / 4)^2 * min(1.0, rho(x, y) / rho_0)
This prevents the proximal operator from clipping dense crowd mass while retaining strong sparse noise suppression.

### R2. Anscombe Variance-Stabilizing Power Formulation in SIRT Adjoint
Under Poisson-distributed point processes, the variance of measurement error scales with count Var(b) proportional to mu. The SIRT solver must operate in the Anscombe variance-stabilized domain:
T(y) = 2 * sqrt(y + 3/8)
where measurement residuals have asymptotically constant variance sigma^2 approx 1, eliminating the gradient starvation on dense clusters that caused v29's -29.95 negative bias.

### R3. Preserved Convex Loss Core & Strict Bitwise v19 Baseline Parity
All core losses must maintain bitwise mathematical equivalence to v19 on Stride 4:
- Mass-Weighted Cell Allocation Loss (lambda_cell = 0.5, alpha = 2.0, gamma = 1.25)
- Flat-DM16 Optimal Transport Loss (lambda_alloc = 1.0, kappa = 20.0)
- Canonical Batch-Wide Occupied Truncated NB NLL Loss (lambda_trunc = 0.20)
- Physical Scale Alignment Loss with Monotonic Background Masking (lambda_scale = 0.05)
- Density-Gated Curvature Power Loss (lambda_curv = 0.50)

### R4. Controlled Two-Loop Ablation Suite
The team must build and execute a clean 6-model single-variable ablation suite:
1. rmr_v30_step0_v19_anchor: Validated bitwise v19 anchor (verifying 72.84 reproduction).
2. rmr_v30_h1_anscombe_sirt: SIRT solver with Anscombe variance stabilization on Stride 4.
3. rmr_v30_h2_dual_lattice_dcsr: Density-conditioned dual lattice (Stride 2/4) with adaptive proximal tau.
4. rmr_v30_h3_anscombe_dual_lattice: Composite of Anscombe VST + Dual-Lattice DCSR (The Primary Sub-60 Hypothesis).
5. rmr_v30_h4_deep_sirt_t8: T=8 unrolling on H3 architecture.
6. rmr_v30_control_no_solver: Direct feedforward baseline (Y_0).

## 3. Acceptance Criteria

### Mathematical Rigor & Code Invariants
- [ ] Trainable parameters strictly <= 105,000 across all 6 model variants (target: <= 104,800).
- [ ] Zero Knowledge Distillation: strictly 0 teacher weights, 0 distillation losses in Stages 1 and 2.
- [ ] Monolith Prevention: All files in rmr_core/ and rmr_v3/ strictly <= 450 lines.
- [ ] Support Invariance: y_0(u) = 0 ==> Radon-Nikodym update remains 0 on true background.
- [ ] 100% test pass rate across tests/rmr_v29/, tests/rmr_v3/, tests/core/.

### Empirical Validation on Canonical ShanghaiTech Part A
- [ ] rmr_v30_step0_v19_anchor reproduces 72.84 +- 0.3 MAE.
- [ ] rmr_v30_h3_anscombe_dual_lattice breaks the Sub-65 barrier (MAE <= 64.9) and targets Sub-60 (MAE < 60.0).
- [ ] Sparse MAE (count < 100) retains the v29 breakthrough: MAE <= 14.0.
- [ ] Dense MAE (count >= 500) improves from 122.71 to <= 105.0.
- [ ] Net validation count bias stays bounded within [-8.0, +3.0] (eliminating v29 H2's -29.95 bias).

## 2026-09-20T09:07:36Z

Full Research Team (1 Orchestrator, 1 Math-Physics Lead, 1 Architecture Engineer, 1 Benchmark Auditor)

Perform an exhaustive, empirical, and mathematical cross-generational analysis of all historical RMR experimental versions (v4 through v30) in `runs/sha_a`, identifying the governing mathematical physics, architectural breakthroughs, and optimization failure modes on ShanghaiTech Part A to synthesize a definitive, parameter-budgeted Sub-50 MAE blueprint.

Working directory: f:/lightweightcrcn
Integrity mode: development

## Requirements

### R1. Comprehensive Cross-Generational Data Aggregation
Extract and tabulate validation metrics (MAE, RMSE, NAE, Bias, Sparse <100, Moderate 100-500, Dense >500, GAME 0-3), training loss dynamics, convergence epochs, parameter counts, and trajectory diagnostics (solver help/harm fractions, energy monotonicity, dispersion saturation) across all completed runs in `runs/sha_a` from v4 to v30.

### R2. Mechanistic Inflection Point & Failure Autopsy
Perform an objective scientific autopsy identifying the exact governing mechanism for every major breakthrough and regression:
- **v19 Canonical (72.39 / 72.84 MAE):** Macro-isotropic regional Poisson modeling, BB-1 adaptive step size, and Morozov discrepancy stopping.
- **v20-v23 (80-90+ MAE):** High-density sample loss scaling instability, micro-attention parameter inflation, and flux drift.
- **v24-v26 (74-76 MAE):** Alternating Barzilai-Borwein (ABB) vs pure BB-1 Rayleigh contraction dynamics, and micro-perspective elevation.
- **v27-v28 (80-100+ MAE):** Multi-variable shotgun configuration changes, cyclic BB instability, and curvature alpha initialization sensitivity.
- **v29 (75-80 MAE):** Sub-pixel Stride 2 coordinate misalignment and Dirac area-scale scaling bugs.
- **v30 (75.81 Anchor vs 15.20 Sparse / 183.49 Dense):** Why Stride 2 Dual-Lattice excels on sparse crowds (15.20 MAE) but collapses on dense crowds (183.49 MAE), and why Anscombe VST destabilizes background cells.

### R3. Determinism & Engineering Architecture Verification
Verify that all code and experiment runner configurations satisfy strict production-grade engineering constraints:
- Deterministic execution auto-enabled by default (`seed_worker`, `torch.Generator`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- Line count ceiling <= 450 lines per Python source file across all packages (`rmr_v3/`, `rmr_core/`).
- Parameter budget <= 105,000 trainable parameters.
- Zero Knowledge Distillation / Zero external teachers.

### R4. Definitive Sub-50 MAE Synthesis Blueprint
Synthesize the cumulative empirical evidence into a mathematically proven, parameter-budgeted architectural blueprint for the next generation:
- Decoupled / Density-Adaptive Multi-Lattice Resolution (Stride 2 for sparse heads, Stride 4 for dense clumps).
- Density-Gated Anscombe VST (active only for high-density counts y > tau_dense to prevent background noise explosion).
- Edge-Preserving / Anisotropic SIRT diffusion to prevent over-smoothing at deeper iterations.

## Acceptance Criteria

### Comprehensive Benchmark Audit Matrix
- [ ] Complete comparative markdown table of all versions in `runs/sha_a` (v4 through v30) with exact numerical metrics extracted from raw `summary.json` and `train_log.csv`.
- [ ] Stratification across density tiers: Sparse (<100), Moderate (100-500), and Dense (>500).

### Mathematical & Physical Rigor
- [ ] Exact analytical formulas and gradient derivations explaining the success or failure of each mechanism.
- [ ] Empirical correlation between diagnostic metrics (`energy_monotonic_fraction`, `solver_help_fraction`, `dispersion_p90`) and validation MAE.

### Next-Generation Specification
- [ ] Actionable, parameter-counted architectural design (<= 105,000 parameters) ready for implementation.
- [ ] Clear ablation ladder testing one single variable at a time (Two-Loop Autoresearch protocol).

