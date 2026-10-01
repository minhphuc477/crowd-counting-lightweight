# Test Infrastructure Specification: RMR-v26 Ultra-Lightweight Crowd Counting

## 1. Executive Summary & Testing Philosophy

The test infrastructure for **RMR-v26 (Regional Measure Reconciliation v26)** establishes a requirement-driven, opaque-box end-to-end verification framework. Under strict project constraints (parameter budget $\le 105,000$, target 104,505 parameters, stride-4 spatial resolution, zero knowledge distillation, and zero teacher models), RMR-v26 addresses the core research question:
> *"Can known discrete regional-count operators replace learned contextual reasoning in ultra-lightweight models?"*

The test suite enforces mathematical rigor, algorithmic stability, and empirical accuracy through an exhaustive **4-Tier Test Case Design Methodology**.

### Core Principles:
1. **Opaque-Box Requirement Verification**: Tests validate public interfaces, mathematical contracts, physical invariants, and macroscopic behavior without coupling to non-contract internal implementation details.
2. **Progressive Testability**: Tests are designed to execute cleanly and provide unambiguous signals across development milestones (M0 through M5), isolating completed components and signaling pending integrations.
3. **Test Independence & Determinism**: Every test case initializes its own state, seeds pseudo-random generators where appropriate, operates without test execution order dependencies, and produces deterministic outcomes.
4. **Authoritative Mathematical Oracles**: Where reference implementations are required (such as MPE-v2 normalized carrier projection and Rayleigh contraction quotients), authoritative mathematical oracles verify exact numerical adherence.

---

## 2. 4-Tier Test Architecture Matrix

The test suite spans all 7 core features defined in `PROJECT.md`, organized into four tiers with a total of **85 test cases** (exceeding the requirement of $\ge 82$ test cases):

| Tier | Focus | Requirement Target | Implemented Cases | Target Scope |
|------|-------|--------------------|-------------------|--------------|
| **Tier 1** | Feature Coverage | $\ge 5$ per feature ($\ge 35$ total) | **36 test cases** | Full coverage of nominal behaviors across F1–F7 |
| **Tier 2** | Boundary & Corner Cases | $\ge 5$ per feature ($\ge 35$ total) | **35 test cases** | Extremes, singularities, zeros, prime dims, stress |
| **Tier 3** | Pairwise Combinations | $\ge 7$ total | **8 test cases** | Cross-feature interactions and joint dynamics |
| **Tier 4** | Real-World Workloads | $\ge 5$ total | **6 test cases** | End-to-end training, TTA inference, dense/sparse benchmarks |
| **Total** | **Comprehensive Suite** | **$\ge 82$ test cases** | **85 test cases** | Complete system integrity and empirical validation |

---

## 3. Feature Mapping & Coverage Catalog

### Feature F1: MPE-v2 Perspective Elevation
- **Contract**: Normalized 1D elevation modulation field preserving vertical mass:
  $$M(v) = 1.0 + \tanh(W v + b)$$
  $$\bar{M} = \frac{1}{H} \sum_{i=1}^H M(v_i)$$
  $$\tilde{P}_4(v) = P_4(v) \cdot \frac{M(v)}{\bar{M}}$$
- **Invariant**: $\frac{1}{H}\sum_{i=1}^H \left(\frac{\tilde{P}_4(v_i)}{P_4(v_i)}\right) \equiv 1.000000 \pm 10^{-6}$ for any non-zero carrier $P_4$.
- **Parameters**: Exactly 64 trainable parameters (`Linear(1, 32)`: 32 weights, 32 biases) with identity warm-start ($W=0, b=0$).
- **Test Allocation**:
  - *Tier 1*: Parameter count (exact 64), identity warm-start, strict mass conservation, horizontal column invariance, smooth gradient backpropagation.
  - *Tier 2*: Minimal height ($H=2$), large height ($H=1024$), prime heights ($H=73, 137$), extreme saturation ($W=100, b=-50$), delta-spike vs uniform carrier.

### Feature F2: Pure BB-1 Trust Damping
- **Contract**: Pure Barzilai-Borwein Rayleigh quotient step size:
  $$\alpha_1 = \frac{\langle s_k, r_k \rangle}{\|r_k\|^2 + \epsilon}$$
  clamped strictly in $[0.5 \cdot \omega_0, 1.2 \cdot \omega_0]$.
- **Invariants**: Disabling alternating BB-2 in canonical mode, non-negativity preservation ($y_t \ge 0$), bounded solver residuals across $T=6$ iterations.
- **Test Allocation**:
  - *Tier 1*: Rayleigh quotient calculation, $[0.5\omega_0, 1.2\omega_0]$ damping clamp, BB-2 exclusion, iterate non-negativity, convergence residual monotonicity.
  - *Tier 2*: Vanishing residual ($r_k \to 0$), orthogonal displacement ($\langle s, r \rangle = 0$), negative curvature ($\langle s, r \rangle < 0$), infinite step demand ($\alpha_1 \to \pm \infty$), single iteration ($T=1$).

### Feature F3: Curvature Loss Damping
- **Contract**: Auxiliary curvature loss locked to $\lambda_{\text{curv}} = 0.0$ in canonical model, relying on the proven convex loss triad:
  - Mass-Weighted Cell L1 loss ($\lambda=0.5, \alpha=2.0, \gamma=1.25$)
  - Flat-DM16 Optimal Transport loss ($\lambda=1.0, \kappa=20.0$)
  - Scale-Balanced Regional NB NLL loss ($\lambda=0.20$)
- **Test Allocation**:
  - *Tier 1*: $\lambda_{\text{curv}} = 0.0$ canonical enforcement, Cell L1 convexity and monotonicity, Flat-DM16 dispersion $\kappa=20$, Regional NB scale-balancing, composite loss gradient flow.
  - *Tier 2*: All-background zero density ($y_{\text{gt}}=0$), hyper-dense single-cell clump ($y > 2000$), zero predicted density epsilon safety, $\lambda_{\text{curv}}=0.10$ boundary behavior, box boundary scaling ($1\times 1$ up to $H\times W$).

### Feature F4: 6-Model Suite Configs
- **Contract**: Matrix of 6 configurations cleanly isolating model components:
  1. `rmr_v26_canonical`: MPE-v2 + Pure BB-1 + Morozov 0.75 + $\lambda_{\text{curv}} = 0.0$.
  2. `rmr_v26_ablation_no_elevation`: `use_perspective_elevation: false` (104,441 params).
  3. `rmr_v26_ablation_no_bb`: `use_barzilai_borwein: false`, fixed $\omega=1.0$.
  4. `rmr_v26_ablation_no_morozov`: `morozov_gamma: 0.0`.
  5. `rmr_v26_ablation_with_curv01`: `lambda_curvature: 0.10`.
  6. `rmr_v26_control_no_solver`: `enable_solver: false` (feedforward $Y_0$).
- **Test Allocation**:
  - *Tier 1*: Structure and parameter validation across all 6 model configurations.
  - *Tier 2*: Missing required keys rejection, unknown hyperparameter rejection, inverted damping clamps, boundary Morozov $\gamma \in \{0.0, 5.0\}$, configuration immutability.

### Feature F5: Math & Integrity Invariants
- **Contract**:
  - Hard parameter ceiling: $\le 105,000$ trainable parameters across all configurations.
  - Radon-Nikodym measure support absorption: $\text{supp}(y_{t+1}) \subseteq \text{supp}(y_t)$ with zero background phantom leakage.
  - Adjoint scale invariance: $H_\nu \mathbf{1} = \mathbf{1}$.
  - Bayesian Morozov discrepancy deadband: $|r| \le \gamma \sigma_b \implies \text{shrinkage to } 0$.
  - Prior analytical bias calibration: $b_0 = \ln(\exp(0.015763) - 1) \approx -4.1422$.
- **Test Allocation**:
  - *Tier 1*: 105k parameter ceiling, Radon-Nikodym support absorption, scale invariance, Morozov deadband shrinkage, prior analytical bias calibration.
  - *Tier 2*: Boundary 105,000 vs 105,001 param stress, zero carrier mass preservation, single-pixel delta support absorption, checkerboard zero preservation, infinite dispersion limit.

### Feature F6: Unit Test & E2E Test Suite
- **Contract**: Complete regression and E2E test suite reliability, test isolation, reproducibility, AMP fp16 compatibility, and bounded resource usage.
- **Test Allocation**:
  - *Tier 1*: Test isolation without shared mutable state, deterministic reproducibility under random seed, AMP float16 precision stability, batch-active foreground normalization, bounded memory scaling during 6 unrolled iterations.
  - *Tier 2*: Single image batch ($B=1$), multi-sample batch ($B=8$), zero learning rate parameter freezing, NaN/Inf gradient detection, empty region handling.

### Feature F7: ShanghaiTech Part A Eval
- **Contract**: Canonical benchmark evaluation:
  - Strict 300 train / 182 test sample enforcement (Zero Ad-hoc Split Policy).
  - Density stratification: Sparse ($gt \le 100$), Moderate ($100 < gt \le 500$), Dense ($gt > 500$).
  - Physical metrics: MAE, RMSE, Bias, and Horizontal Flip TTA ($0.5(Y + \text{Flip}_H(Y_{\text{flip}}))$).
  - Acceptance thresholds: Overall MAE $\le 70.0$, Sparse MAE $\le 18.0$, Dense MAE $\le 125.0$, Bias $\in [-5.0, +2.0]$.
- **Test Allocation**:
  - *Tier 1*: Zero ad-hoc split enforcement, density bin stratification, metric calculation accuracy, TTA averaging correctness, acceptance criteria threshold validators.
  - *Tier 2*: All-zero ground truth evaluation, single-sample evaluation ($N=1$), identical predictions ($MAE=0, Bias=0$), constant offset bias isolation, bin boundary edge counts ($C=100.0, 500.0$).

---

## 4. Test Directory Layout & Organization

```
f:\lightweightcrcn\
├── TEST_INFRA.md                          # Test infrastructure specification (this document)
├── TEST_READY.md                          # Test suite readiness declaration
└── tests\
    └── e2e_v26\
        ├── __init__.py                    # Package initialization
        ├── conftest.py                    # Shared fixtures, oracles, synthetic generators, config builders
        ├── test_tier1_feature_coverage.py # Tier 1: 36 Feature Coverage test cases (F1–F7)
        ├── test_tier2_boundary_corner.py  # Tier 2: 35 Boundary & Corner test cases (F1–F7)
        ├── test_tier3_pairwise_combinations.py # Tier 3: 8 Pairwise Cross-Feature test cases
        └── test_tier4_workload_scenarios.py   # Tier 4: 6 Real-World Workload Scenarios
```

---

## 5. Execution Commands & Test Harness

### Running the Entire E2E Test Suite:
```bash
.venv\Scripts\pytest.exe tests/e2e_v26/ -v
```

### Running Specific Tiers:
```bash
# Run Tier 1: Feature Coverage
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier1_feature_coverage.py -v

# Run Tier 2: Boundary & Corner Cases
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier2_boundary_corner.py -v

# Run Tier 3: Cross-Feature Pairwise Combinations
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier3_pairwise_combinations.py -v

# Run Tier 4: Real-World Workload Scenarios
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier4_workload_scenarios.py -v
```

### Progressive Testability Behavior:
- When running in **Milestone M0** (prior to M1/M2/M3 completion), tests assert specifications and contracts against authoritative mathematical oracles and active model capabilities.
- Tests targeting disk-based YAML configs (`configs/rmr_v26/`) cleanly signal `pytest.skip("Pending M3")` when files have not yet been placed on disk, while validating the in-memory schema and parameters immediately.
- Once M1, M2, and M3 implementers deliver their components, all tests automatically switch to verifying the integrated production code.
