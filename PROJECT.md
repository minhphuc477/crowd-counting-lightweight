# Project: RMR-v26 Ultra-Lightweight Crowd Counting

## Architecture
RMR-v26 (Regional Measure Reconciliation v26) is an ultra-lightweight crowd counting framework strictly constrained to $\le 105,000$ trainable parameters (target: 104,505 parameters), operating on stride-4 spatial resolution with zero knowledge distillation and zero teacher models.

### Key Architectural Components:
1. **Backbone**: Truncated MobileNetV4 (`MobileNetV4Backbone`, 87,568 params) producing multi-scale feature representations at strides $(4, 8, 16)$.
2. **Neck**: ASPP-Lite FPN Neck (`ASPPLiteFPNNeck`, width 32, dilations 1, 3, 6 + GAP, 10,784 params) unifying multi-scale contexts into fine carrier $P_4$.
3. **MPE-v2 Perspective Elevation**: MicroPerspectiveElevation v2 (exactly 64 params, identity warm-start) modulating $P_4$ with strictly mass-conserved vertical weighting:
   $$M(v) = 1.0 + \tanh(W v + b)$$
   $$\bar{M} = \frac{1}{H} \sum_{i=1}^H M(v_i)$$
   $$\tilde{P}_4(v) = P_4(v) \cdot \frac{M(v)}{\bar{M}}$$
   satisfying $(1/H) \sum_{i=1}^H (\tilde{P}_4(v_i) / P_4(v_i)) \equiv 1.000000 \pm 10^{-6}$.
4. **Heads**:
   - `FineMeasureHead` (1,475 params): Projects $\tilde{P}_4$ to initial continuous density measure $y_0$ with softplus and analytical bias $b_0 \approx -4.1422$.
   - `ProbabilisticRegionalEvidenceHead` (4,131 params): Predicts multi-scale regional counts $\mu_r$, dispersion $\kappa_r$, and zero-inflation hurdle $\pi_r$.
   - `ScaleRoutingHead` (483 params): Predicts continuous regional scale assignments across [32, 64, 128] px.
5. **Unrolled SIRT Solver**:
   - $T=6$ unrolled iterations solving the continuous-discrete measure reconciliation inverse problem.
   - Pure BB-1 Rayleigh contraction step size: $\alpha_1 = \langle s_k, r_k \rangle / (\|r_k\|^2 + \epsilon)$.
   - Trust damping clamp strictly bounded to $[0.5 \cdot \omega_0, 1.2 \cdot \omega_0]$.
   - Radon-Nikodym measure adjoint field guaranteeing support absorption ($\text{supp}(y_{t+1}) \subseteq \text{supp}(y_t)$) and scale invariance ($H_\nu \mathbf{1} = \mathbf{1}$).
   - Bayesian Morozov discrepancy deadband with $\gamma = 0.75$ preventing measurement noise amplification.
6. **Loss Core**:
   - Locked auxiliary curvature loss $\lambda_{\text{curv}} = 0.0$ in canonical model.
   - Convex loss triad: Mass-Weighted Cell L1 ($\lambda=0.5, \alpha=2.0, \gamma=1.25$), Flat-DM16 Optimal Transport ($\lambda=1.0, \kappa=20.0$), and Scale-Balanced Regional NB ($\lambda=0.20$).

---

## Feature Inventory
Every feature from the Survey phase is mapped to an assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | MPE-v2 Modulation | Normalized 1D elevation modulation field preserving vertical mass $(1/H)\sum (\tilde{P}_4 / P_4) == 1.0 \pm 1e-6$, exactly 64 params, identity warm-start | M1 | ORIGINAL_REQUEST §2.R1 |
| F2 | Pure BB-1 Trust Damping | Rayleigh contraction step size clamped in $[0.5\omega_0, 1.2\omega_0]$ | M2 | ORIGINAL_REQUEST §2.R2 |
| F3 | Curvature Loss Damping | Lock $\lambda_{\text{curv}} = 0.0$ in canonical, preserve v19 convex loss triad | M3 | ORIGINAL_REQUEST §2.R3 |
| F4 | 6-Model Suite Configs | Complete 6-model matrix configs: canonical, no_elevation, no_bb, no_morozov, with_curv01, control_no_solver | M3 | ORIGINAL_REQUEST §2.R4 |
| F5 | Math & Integrity Invariants | Param count $\le 105,000$ (target 104,505), mass conservation, Radon-Nikodym support absorption, zero phantom counts | M1, M2, M0 | ORIGINAL_REQUEST §3 |
| F6 | Unit Test & E2E Test Suite | 100% unit test pass rate across regression and new v26 test suites; 4-tier E2E test suite | M0, M4 | ORIGINAL_REQUEST §3 |
| F7 | ShanghaiTech Part A Eval | Canonical 182 test images, TTA, stride 4, MAE $\le 70.0$, Sparse $\le 18.0$, Dense $\le 125.0$, Bias $\in [-5.0, +2.0]$ | M4, M5 | ORIGINAL_REQUEST §3 |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Requirement-driven 4-tier E2E test suite, test harness, `TEST_INFRA.md`, and `TEST_READY.md` | none | IN_PROGRESS |
| M1 | MPE-v2 Architecture | Implement MPE-v2 in `rmr_v3/model.py`, verify 64 params, identity warm-start, strict mass conservation invariant | none | IN_PROGRESS |
| M2 | Pure BB-1 Trust Damping | Implement BB-1 $[0.5\omega_0, 1.2\omega_0]$ trust damping in `rmr_v3/solver.py`, parameterize config options | M1 | PLANNED |
| M3 | Loss Core & 6-Model Configs | Create `configs/rmr_v26/` with 6 model configurations locking $\lambda_{\text{curv}}=0.0$ on canonical and setting up ablation variants | M1, M2 | PLANNED |
| M4 | Model Suite Verification | Train/validate and run regression testing across the 6-model suite, verify parameter budgets and loss logging | M3, M0 | PLANNED |
| M5 | Final Acceptance Benchmark | Execute canonical ShanghaiTech Part A evaluation (182 test images, TTA, stride 4), verify all acceptance criteria (MAE $\le 70.0$, Sparse $\le 18.0$, Dense $\le 125.0$, Bias $\in [-5.0, +2.0]$), followed by Phase 2 adversarial coverage hardening | M4, M0 | PLANNED |

---

## Interface Contracts

### MPE-v2 (`MicroPerspectiveElevation`)
- Location: `rmr_v3/model.py`
- Input: `x: torch.Tensor` of shape `[B, C, H, W]` or `[C, H, W]`
- Output: `torch.Tensor` of identical shape and dtype
- Parameter count: exactly 64 (`Linear(1, 32)`: 32 weights, 32 biases)
- Math invariant: $(1/H) \sum_{i=1}^H (\tilde{x}_{b, c, i, j} / x_{b, c, i, j}) \equiv 1.000000 \pm 1e-6$
- Warm-start: `W = 0.0, b = 0.0` -> $\tilde{x} = x$

### BB-1 Solver (`unrolled_sirt_solver`)
- Location: `rmr_v3/solver.py`
- Arguments: `bb_clamp_min: float = 0.5`, `bb_clamp_max: float = 1.2`, `effective_omega: float = 1.0`
- Clamping behavior: `current_omega = torch.clamp(omega_candidate, min=bb_clamp_min * effective_omega, max=bb_clamp_max * effective_omega)`
- Invariant: When `use_barzilai_borwein=False`, `current_omega = effective_omega`

### Config Contracts (`configs/rmr_v26/`)
- All models must adhere to `RMRv3Config` schema in `rmr_v3/config.py`.
- Canonical: `use_perspective_elevation: true`, `use_barzilai_borwein: true`, `bb_clamp_min: 0.5`, `bb_clamp_max: 1.2`, `morozov_gamma: 0.75`, `lambda_curvature: 0.0`.
- Control and ablations isolate each component cleanly without touching other hyperparameters.

---

## Code Layout
- `rmr_v3/model.py`: Model architecture, `MicroPerspectiveElevation` v2 implementation
- `rmr_v3/solver.py`: Unrolled SIRT solver, BB-1 Rayleigh contraction step size and clamping
- `rmr_v3/config.py`: Config parsing and allowed keys validation
- `configs/rmr_v26/`: 6 configuration YAML files
- `tests/rmr_v26/`: Dedicated unit test suite for v26 features and invariants
- `tests/e2e_v26/`: Opaque-box E2E test suite (Tiers 1-4)
- `tools/`: Benchmark comparison and evaluation scripts
