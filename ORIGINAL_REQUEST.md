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
