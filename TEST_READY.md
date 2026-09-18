# Test Readiness Declaration: RMR-v26 Ultra-Lightweight Crowd Counting

**Milestone**: M0 (E2E Test Suite Creation & Verification)  
**Date**: 2026-09-18  
**Author**: `test_writer_m0`  
**Test Suite Root**: `tests/e2e_v26/`  
**Status**: **READY & VERIFIED (PASS: 97, SKIP: 1, FAIL: 0, TOTAL: 98)**  

---

## 1. Executive Summary

The opaque-box, requirement-driven E2E test suite for **RMR-v26 (Regional Measure Reconciliation v26)** has been implemented and verified. All requirements specified in `PROJECT.md`, `ORIGINAL_REQUEST.md`, and the survey handoffs have been systematically codified across a **4-Tier Test Case Design Methodology**.

The test suite contains **98 total test cases** (97 passed, 1 skipped pending disk configs creation in M3), substantially exceeding the project minimum threshold of $\ge 82$ test cases.

---

## 2. 4-Tier Test Suite Summary & Execution Results

| Tier | Test File | Target | Implemented | Passed | Skipped | Failed | Execution Time |
|------|-----------|--------|-------------|--------|---------|--------|----------------|
| **Tier 1: Feature Coverage** | `tests/e2e_v26/test_tier1_feature_coverage.py` | $\ge 35$ ($\ge 5$ / feat) | 43 | 42 | 1* | 0 | 2.48s |
| **Tier 2: Boundary & Corner Cases** | `tests/e2e_v26/test_tier2_boundary_corner.py` | $\ge 35$ ($\ge 5$ / feat) | 41 | 41 | 0 | 0 | 1.54s |
| **Tier 3: Pairwise Combinations** | `tests/e2e_v26/test_tier3_pairwise_combinations.py` | $\ge 7$ | 8 | 8 | 0 | 0 | 1.63s |
| **Tier 4: Real-World Workloads** | `tests/e2e_v26/test_tier4_workload_scenarios.py` | $\ge 5$ | 6 | 6 | 0 | 0 | 3.60s |
| **TOTAL** | `tests/e2e_v26/` | **$\ge 82$** | **98** | **97** | **1** | **0** | **7.14s** |

*\*Note: The single skipped test (`test_f4_on_disk_configs_check`) provides a clean progressive signal pending Milestone M3 when `configs/rmr_v26/*.yaml` files are written to disk. The in-memory schema and parameter budgets for all 6 configurations are actively verified and passing.*

---

## 3. Feature Coverage Matrix (F1 through F7)

| Feature | Scope | Key Invariants Verified | Tier 1 Tests | Tier 2 Tests | Tier 3/4 Coverage |
|---------|-------|-------------------------|--------------|--------------|-------------------|
| **F1: MPE-v2 Perspective Elevation** | `rmr_v3/model.py` | Exact 64 parameters (`Linear(1, 32)`), identity warm-start ($W=0, b=0$), vertical mass conservation $(1/H)\sum (\tilde{P}_4 / P_4) \equiv 1.000000 \pm 10^{-6}$, column invariance, smooth backprop. | 6 tests | 5 tests | T3: 3 tests, T4: 2 tests |
| **F2: Pure BB-1 Trust Damping** | `rmr_v3/solver.py` | Rayleigh contraction quotient $\alpha_1 = \langle s_k, r_k \rangle / (\|r_k\|^2 + \epsilon)$, damping clamp $[0.5\omega_0, 1.2\omega_0]$, exclusion of BB-2 in canonical, iterate non-negativity $y_t \ge 0$, bounded residual fields across $T=6$. | 5 tests | 5 tests | T3: 3 tests, T4: 2 tests |
| **F3: Curvature Loss Damping** | `rmr_v3/losses.py` | $\lambda_{\text{curv}} = 0.0$ in canonical, convex loss triad (Mass-Weighted Cell L1 with $\alpha=2.0, \gamma=1.25$, Flat-DM16 with $\kappa=20.0$, Scale-Balanced Regional NB with $\lambda=0.20$), smooth backpropagation without non-convex gradient spikes. | 5 tests | 5 tests | T3: 2 tests, T4: 1 test |
| **F4: 6-Model Suite Configs** | `configs/rmr_v26/` | Strict schema conformance across all 6 models: canonical, no_elevation, no_bb, no_morozov, with_curv01, control_no_solver; pairwise parameter and hyperparameter isolation. | 7 tests | 5 tests | T3: 2 tests, T4: 1 test |
| **F5: Math & Integrity Invariants** | System-wide | Parameter budget ceiling $\le 105,000$ (canonical = 104,505, no_elevation = 104,441), Radon-Nikodym support absorption ($\text{supp}(y_{t+1}) \subseteq \text{supp}(y_t)$), scale invariance $H_\nu \mathbf{1} = \mathbf{1}$, Morozov deadband ($\gamma=0.75$), analytical bias $b_0 \approx -4.1422$. | 9 tests | 5 tests | T3: 3 tests, T4: 1 test |
| **F6: Unit Test & E2E Reliability** | Harness & Runtime | Test isolation without mutable contamination, deterministic reproducibility under random seed, AMP float16 precision stability, batch-active foreground normalization, bounded memory scaling. | 5 tests | 5 tests | T3: 1 test, T4: 2 tests |
| **F7: ShanghaiTech Part A Eval** | `rmr_core/` | Zero Ad-hoc Split Policy (300 train / 182 test), density stratification (sparse $\le 100$, moderate $100-500$, dense $> 500$), MAE, RMSE, Bias computation accuracy, horizontal flip TTA averaging, acceptance criteria threshold validators. | 6 tests | 5 tests | T3: 1 test, T4: 2 tests |

---

## 4. How to Execute the Test Suite

Run all tests via pytest in the repository root:

```bash
# Run entire E2E test suite (all 4 tiers)
.venv\Scripts\pytest.exe tests/e2e_v26/ -v

# Run individual tiers
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier1_feature_coverage.py -v
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier2_boundary_corner.py -v
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier3_pairwise_combinations.py -v
.venv\Scripts\pytest.exe tests/e2e_v26/test_tier4_workload_scenarios.py -v
```

---

## 5. Milestone Progressive Readiness & Downstream Handoff

The test suite serves as the contract and acceptance gate for the upcoming implementation milestones:
1. **Milestone M1 (MPE-v2 Architecture in `rmr_v3/model.py`)**:
   Implement vertical normalization: $M(v) = 1.0 + \tanh(W v + b)$, $\bar{M} = (1/H)\sum M(v)$, $\tilde{P}_4 = P_4 \cdot (M / \bar{M})$.
   Verification gate: `test_f1_mpe_v2_model_class_mass_conservation` will verify the model class directly.
2. **Milestone M2 (Pure BB-1 Trust Damping in `rmr_v3/solver.py`)**:
   Parameterize `bb_clamp_min=0.5` and `bb_clamp_max=1.2` in `unrolled_sirt_solver`, `RMRv3Config`, and `ALLOWED_MODEL_KEYS`.
   Verification gate: `test_f2_bb1_trust_damping_clamping_bounds` and solver integration tests.
3. **Milestone M3 (Loss Core & 6-Model Configs in `configs/rmr_v26/`)**:
   Create the 6 YAML configuration files locking $\lambda_{\text{curv}} = 0.0$ in canonical.
   Verification gate: `test_f4_on_disk_configs_check` will automatically un-skip and validate all on-disk YAML files.
4. **Milestone M4 & M5 (Verification & Final Acceptance Benchmark)**:
   All regression and E2E suites will run together as the formal release qualification gate.
