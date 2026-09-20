# RMR-v32 Experimental Designs & Scientific Suite Specification

**Author:** DeepCode AI Research Team  
**Date:** 2026-09-21  
**Target Venues:** CVPR / ECCV / IEEE TPAMI  
**Framework:** Continuous-to-Discrete Operator-Theoretic Radon Measure Recovery (RMR)

---

## 1. Overview & Methodological Principles

To meet the empirical and theoretical standards of top-tier artificial intelligence and computer vision publications, RMR-v32 moves beyond ad-hoc heuristics into a **principled 12-experiment scientific matrix**. 

Each experiment is designed under strict scientific protocols:
1. **Single-Variable Hypothesis Testing:** Exactly one mathematical operator or hyperparameter is modified relative to the Step 0 Anchor.
2. **Deterministic Reproducibility:** Fixed seed (42), bitwise identical initialization, canonical dataset partitions (SHA: 300 Train / 182 Test; SHB: 400 Train / 316 Test). Zero ad-hoc splits (no 270/30).
3. **Strict Parameter Ceiling:** For edge regime evaluations, trainable parameters are strictly bounded: $\le 105,000$ (baseline: 104,441; composite: 104,753).
4. **Zero Knowledge Distillation:** Pure standalone capability without external teacher models in native evaluation.

---

## 2. Complete 12-Experiment Scientific Matrix

| # | Config Name | Track | Mathematical Operator / Hypothesis | Params | Role in Paper |
|---|---|---|---|---|---|
| 1 | `rmr_v32_step0_anchor.yaml` | Track A | Golden Baseline: Full v19 replication in clean v3 codebase | 104,441 | Table 1: Main Baseline (72.61 MAE) |
| 2 | `rmr_v32_control_no_solver.yaml` | Track A | Control: Feedforward $y_0$ only ($T=0$, solver disabled) | 104,441 | Table 3: Proves Solver Gain |
| 3 | `rmr_v32_h1_cpcm.yaml` | Track A | H1: Continuous Perspective Carrier Modulation ($M(u,v)$) | 104,753 | Table 2: Sub-50 Candidate (+312 params) |
| 4 | `rmr_v32_h2_floor_suppression.yaml` | Track A | H2: $C^1$ Smooth Floor Suppression ($\tau = 0.008$) | 104,441 | Table 2: Noise Suppression (0 params) |
| 5 | `rmr_v32_h3_composite.yaml` | Track A | H3: Composite CPCM + Smooth Floor | 104,753 | Table 2: Sub-50 Main Model |
| 6 | `rmr_v32_h4_dense_loss_scaling.yaml` | Track A | H4: Calibrated Dense Loss Weighting (thresh=250) | 104,441 | Table 2: Anti-Saturation (0 params) |
| 7 | `rmr_v32_h5_conservative_solver.yaml` | Track A | H5: Zero TV Diffusion ($\lambda_{\text{TV}} = 0$) | 104,441 | Table 3: Peak Preservation (0 params) |
| 8 | `rmr_v32_ablation_static_windowing.yaml` | Track B | Ablation: Static Windowing ($A_\pi^\top \to A_{\text{static}}^\top$) | 103,958 | Table 3: Proves Theorem 2 ($\Delta = -7.84$) |
| 9 | `rmr_v32_ablation_flat_adjoint.yaml` | Track B | Ablation: Naive Back-Projection ($A_{\text{flat}}^\top$) | 104,441 | Table 3: Proves Claim 2 (Radon-Nikodym) |
| 10 | `rmr_v32_ablation_no_morozov.yaml` | Track B | Ablation: Unconstrained Fitting ($\gamma = 0.0$) | 104,441 | Table 3: Proves Theorem 3 (Deadband) |
| 11 | `rmr_v32_ablation_uniform_reliability.yaml` | Track B | Ablation: Unweighted Fitting ($W = I$) | 104,441 | Table 3: Proves SNR Variance Weighting |
| 12 | `rmr_v32_shb_anchor.yaml` | Track C | Benchmark: ShanghaiTech Part B Evaluation | 104,441 | Table 4: Multi-Dataset Generalization |

---

## 3. Detailed Experimental Descriptions & Hypotheses

### Track A: Primary Hypotheses & Baselines

#### Experiment 1: `rmr_v32_step0_anchor.yaml`
- **Theoretical Target:** Reproduce the exact v19 canonical isotropic baseline (72.61 TTA MAE / 72.84 Direct MAE) within the modular `rmr_v3` architecture.
- **Formulation:** 6 unrolled SIRT iterations, BB-1 Rayleigh quotient step sizes, Morozov deadband ($\gamma = 0.75$), SNR reliability weighting, Dynamic Windowing ($A_\pi^\top$).
- **Parameters:** Exactly 104,441 parameters.

#### Experiment 2: `rmr_v32_control_no_solver.yaml`
- **Theoretical Target:** Quantify the exact contribution of unrolled inverse problem solving vs pure feedforward convolutional prediction.
- **Formulation:** $T = 0$, $y = y_0$. Evaluates raw backbone + fine head.
- **Expectation:** MAE degrades from 72.61 to ~82.50, demonstrating that the unrolled SIRT solver provides a $\approx 10.0$ MAE improvement.

#### Experiment 3: `rmr_v32_h1_cpcm.yaml`
- **Theoretical Target:** Test whether continuous 2D coordinate-to-channel modulation $M(u, v) = 1 + \tanh(\text{MLP}(u, v))$ resolves camera perspective foreshortening in dense clumps.
- **Parameters:** Adds 312 parameters (104,753 total). Zero-initialized identity warm-start.
- **Expectation:** Improves dense crowd recovery and reduces spatial distortion along the perspective gradient.

#### Experiment 4: `rmr_v32_h2_floor_suppression.yaml`
- **Theoretical Target:** Eliminate background softplus leakage without suffering from the Dying ReLU trap.
- **Formulation:** $C^1$-continuous quadratic floor suppression:
  $$f(y) = y - 0.5\tau \quad \text{if } y > \tau, \quad \frac{y^2}{2\tau} \quad \text{if } y \le \tau$$
  with $\tau = 0.008$. Gradients remain non-zero: $f'(y) = y / \tau > 0$ for $y \in (0, \tau)$.
- **Expectation:** Suppresses false-positive background noise and reduces GAME metrics.

#### Experiment 5: `rmr_v32_h3_composite.yaml`
- **Theoretical Target:** Evaluate the synergistic combination of CPCM (perspective modulation) and $C^1$ floor suppression.
- **Parameters:** Exactly 104,753 parameters.

#### Experiment 6: `rmr_v32_h4_dense_loss_scaling.yaml`
- **Theoretical Target:** Break the dense crowd optimization saturation bottleneck (Train Dense MAE = 126.67).
- **Statistical Calibration:** Calibrated against empirical 512x512 crop statistics (median = 181, p75 = 357):
  `dense_loss_thresh: 250.0`, `dense_loss_norm: 250.0`, `dense_loss_max_boost: 1.5`.
  Smoothly up-weights the top 33% dense crops up to 2.5x without destabilizing moderate crowds.

#### Experiment 7: `rmr_v32_h5_conservative_solver.yaml`
- **Theoretical Target:** Prevent isotropic TV diffusion from washing point mass from high-density Dirac peaks into surrounding cells.
- **Formulation:** Set $\lambda_{\text{TV}} = 0.0$, preserving peak mass concentration during SIRT iterations.

---

### Track B: Formal Operator Ablations (The Core of the Paper's Table 3)

#### Experiment 8: `rmr_v32_ablation_static_windowing.yaml`
- **Ablation:** Replaces Dynamic Windowing ($A_\pi^\top$) with static uniform multi-scale windows ($A_{\text{static}}^\top$).
- **Theoretical Link:** Directly verifies Theorem 2 (Operator Equivalence under Perspective Geometry). In v19, dynamic windowing accounted for $\Delta = -7.84$ MAE.

#### Experiment 9: `rmr_v32_ablation_flat_adjoint.yaml`
- **Ablation:** Replaces Radon-Nikodym measure modulation ($A_{\text{RN}}^\top(r) = r \odot \frac{y}{\int_R y}$) with naive uniform back-projection ($A_{\text{flat}}^\top(r) = \frac{r}{|R|}$).
- **Theoretical Link:** Directly proves Claim 2: uniform back-projection distributes mass equally across empty and occupied cells, causing severe background blur.

#### Experiment 10: `rmr_v32_ablation_no_morozov.yaml`
- **Ablation:** Sets $\gamma = 0.0$, disabling the Bayesian Morozov discrepancy deadband.
- **Theoretical Link:** Directly verifies Theorem 3: without the Morozov deadband, the solver overfits to neural surrogate noise, amplifying residuals in low-confidence regions.

#### Experiment 11: `rmr_v32_ablation_uniform_reliability.yaml`
- **Ablation:** Sets $W = I$ (all regions weighted equally with $w_R = 1.0$).
- **Theoretical Link:** Quantifies the value of the Poisson-Gamma / Negative-Binomial noise variance weighting $w_R = \frac{\hat{\mu}_R}{\hat{\sigma}_R^2}$.

---

### Track C: Multi-Dataset Benchmark

#### Experiment 12: `rmr_v32_shb_anchor.yaml`
- **Dataset:** ShanghaiTech Part B (400 Train / 316 Test).
- **Target:** Verify cross-dataset transfer and adaptation to sparse street scenes (mean count = 123.2, `init_m0 = 0.002500`).
- **Parameters:** 104,441 parameters.
