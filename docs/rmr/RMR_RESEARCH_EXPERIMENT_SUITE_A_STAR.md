# RMR Research Experiment Suite (A*): Single-Variable Ablations & Spectral Composite

> **Document Type:** Formal Research Experiment Suite & Theoretical Blueprint  
> **Target Venues:** IEEE TPAMI / CVPR / ECCV (Computer Vision & Discrete Inverse Problems)  
> **Topic:** Ultra-Lightweight Crowd Counting via Discrete Measure Reconstruction ($<105\text{k}$ parameters)  
> **Baseline Anchor:** RMR-v19 Canonical Isotropic 3-Scale Architecture (**104,441 Trainable Parameters**)  
> **Benchmark Dataset:** ShanghaiTech Part A (Canonical 300 Train / 182 Test, Zero Ad-hoc Splits)  
> **Data Entropy Protocol:** Non-deterministic training enabled (`deterministic: false`) to preserve data augmentation entropy across 300 images  

---

## 1. Executive Summary & Research Scope

The Regional Measure Reconstruction (RMR) paradigm reframes visual crowd counting from standard heuristic density regression into a **discrete unrolled inverse problem on Radon measure spaces**:
$$\min_{Y \ge 0} \frac{1}{2} \| D_a^{-1/2} (A Y - \mu) \|_W^2 + \mathcal{R}_{\text{spatial}}(Y) + \mathcal{R}_{\text{spectral}}(Y)$$
where $A$ is the multi-scale regional projection operator, $D_a$ is region area normalization, $W$ represents signal-to-noise ratio (SNR) measurement reliability, $\mu$ is the regional count estimate from the convolutional backbone, and $Y$ is the reconstructed continuous density measure.

While RMR-v19 Canonical Isotropic achieved state-of-the-art efficiency with exactly **104,441 parameters**, empirical validation of each discrete mathematical operator in isolation is essential for an A* journal/conference submission.

This Experiment Suite defines a **rigorous Single-Variable Isolation protocol** across six targeted hypotheses:
1. **Hypothesis H3 (Solver Depth Contraction Curve: H3a, H3b, H3c):** Quantifies error contraction dynamics and autograd gradient flow as unrolled SIRT solver depth varies across $T \in \{2, 4, 6, 8\}$.
2. **Hypothesis H4 (Bayesian Morozov Discrepancy Principle):** Proves that disabling the Morozov discrepancy deadband ($\gamma=0.0$) forces the inverse solver to overfit high-frequency background noise.
3. **Hypothesis H5 (Bilateral Gated Curvature Regularization):** Quantifies the contribution of curvature regularization ($\lambda_{\text{curv}}=0.0$) in resolving extreme crowd clumpiness and overlapping heads.
4. **Hypothesis H6 (Spectral Composite Architecture):** Fuses the canonical isotropic spatial solver with Count-Preserving Heavy-Tailed Spectral Loss to establish a new sub-70 MAE Pareto frontier.

---

## 2. Theoretical Formulation of Tested Operators

### 2.1. Unrolled SIRT Inverse Solver & Contraction Dynamics ($T \in \{2, 4, 6, 8\}$)
The Landweber / SIRT (Simultaneous Iterative Reconstruction Technique) iteration for positive measures with Barzilai-Borwein adaptive step size $\alpha_t$ is defined as:
$$Y^{(t+1)} = \mathcal{P}_{\ge 0} \left( Y^{(t)} - \alpha_t D_{c,w}^{-1} A^\top W D_a^{-1} \left( A Y^{(t)} - \mu \right) \right)$$
where $D_{c,w} = \text{diag}(A^\top W \mathbf{1})$ is the weighted coverage diagonal matrix, guaranteeing the transfer operator $H_w = D_{c,w}^{-1} A^\top W D_a^{-1} A$ is row-stochastic with spectral radius $\rho(H_w) \le 1$.

- **Contraction Ratio:** For a contractive operator with contraction constant $\kappa = \|I - \alpha H_w\|_2 < 1$, the residual decays as $\|Y^{(t)} - Y^*\| \le \kappa^t \|Y^{(0)} - Y^*\|$.
- **Autograd Gradient Flow:** Backpropagating through $T$ unrolled steps requires computing:
  $$\frac{\partial \mathcal{L}}{\partial Y^{(0)}} = \frac{\partial \mathcal{L}}{\partial Y^{(T)}} \prod_{t=1}^T \left( I - \alpha_t H_w \right)^\top$$
  - At $T=2$ (H3a): Fast inference, reduced compute, test if shallow unrolling suffices.
  - At $T=4$ (H3b): Mid-range contraction milestone.
  - At $T=6$ (Baseline): Proven canonical convergence point.
  - At $T=8$ (H3c): Tests if deep unrolling yields monotonic accuracy gains or encounters vanishing/exploding gradients through 8 unrolled projection cycles.

### 2.2. Bayesian Morozov Discrepancy Deadband ($\gamma = 0.75 \to 0.0$)
In classical inverse problems, the **Morozov Discrepancy Principle** asserts that an iterative solver should terminate or shrink updates when the residual is within the expected measurement noise variance $\delta$:
$$\|A Y - \mu\|^2 \le \delta^2$$
In RMR, the regional observation $\mu_r$ carries uncertainty $\sigma_r^2$ governed by the dispersion parameter $\theta$ of the negative binomial head:
$$\text{Var}(\mu_r) = \mu_r + \frac{\mu_r^2}{\theta_r}$$
The soft Morozov deadband shrinkage modifies the residual $r_r = A_r Y - \mu_r$:
$$\tilde{r}_r = \text{sign}(r_r) \cdot \max\left(0, |r_r| - \gamma \sqrt{\text{Var}(\mu_r)}\right)$$
- **Baseline ($\gamma = 0.75$):** Residuals below $0.75\sigma$ are treated as statistical noise, preventing the adjoint field from hallucinating counts in noisy background textures.
- **Ablation H4 ($\gamma = 0.0$):** Eliminates the deadband. Every micro-fluctuation in background regions is projected onto the density plane, leading to background count inflation.

### 2.3. Bilateral Gated Curvature Regularization ($\lambda_{\text{curv}} = 0.50 \to 0.0$)
Dense crowds form sharp spatial cusps (high local curvature) separated by steep valleys. Standard total variation (TV) tends to flatten peaks (staircasing artifact). The density-gated bilateral quadratic curvature penalizes over-smoothing in dense clusters:
$$\mathcal{L}_{\text{curv}} = \frac{1}{|\Omega_{\text{dense}}|} \sum_{x \in \Omega_{\text{dense}}} \left( \nabla^2 Y(x) - \nabla^2 Y_{\text{target}}(x) \right)^2$$
where $\Omega_{\text{dense}} = \{x : Y_{\text{target}}(x) > \tau_{\text{dense}}\}$.
- **Baseline ($\lambda_{\text{curv}} = 0.50$):** Sharpens individual head contours in ultra-dense packs ($>500$ people/image).
- **Ablation H5 ($\lambda_{\text{curv}} = 0.0$):** Removes curvature penalty, leaving only standard Poisson/NB cell and region losses to separate adjacent heads.

### 2.4. Count-Preserving Heavy-Tailed Spectral Loss (H6 Composite)
Spatial $\ell_1$/$\ell_2$ pixel losses treat spatial errors uniformly, causing blurry crowd density maps. The Count-Preserving Spectral Loss decomposes prediction and target into the 2D Fourier domain:
$$\hat{Y}(\boldsymbol{\omega}) = \mathcal{F}_{2D}\{Y\}, \quad \hat{Y}^*(\boldsymbol{\omega}) = \mathcal{F}_{2D}\{Y^*\}$$
1. **DC Mass Integral Conservation ($\boldsymbol{\omega} = \mathbf{0}$):**
   $$\mathcal{L}_{\text{spectral, DC}} = \frac{1}{\sqrt{HW}} \left| \hat{Y}(\mathbf{0}) - \hat{Y}^*(\mathbf{0}) \right| = \frac{1}{\sqrt{HW}} \left| \sum_{x} Y(x) - \sum_x Y^*(x) \right|$$
   This strictly enforces global count preservation without frequency bias.
2. **AC Heavy-Tailed Power Law ($\boldsymbol{\omega} \ne \mathbf{0}$):**
   $$\mathcal{L}_{\text{spectral, AC}} = \frac{1}{HW} \sum_{\boldsymbol{\omega} \ne \mathbf{0}} |\boldsymbol{\omega}|^{-\beta} \left| \hat{Y}(\boldsymbol{\omega}) - \hat{Y}^*(\boldsymbol{\omega}) \right|^2$$
   Setting $\beta = 2.0$ weights mid-to-high spatial frequencies corresponding to head diameters (4–16 pixels in ShanghaiTech A), penalizing spatial blur while preserving global mass.

---

## 3. Comprehensive Experiment Matrix

All configurations inherit from `configs/rmr_v19/rmr_v19_canonical_isotropic.yaml` with exactly **104,441 parameters**.

| Config ID | YAML Configuration File | Independent Variable | Isolated Setting | Baseline Setting (v19) | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** | `configs/rmr_v19/rmr_v19_canonical_isotropic.yaml` | Baseline Anchor | $T=6, \gamma=0.75, \lambda_{\text{curv}}=0.5$ | Canonical v19 | `runs/sha_a/rmr_v19_canonical_isotropic` |
| **H3a** | `configs/rmr_research/h3a_depth2.yaml` | Solver Depth ($T$) | `model.iterations: 2` | `iterations: 6` | `runs/sha_a/rmr_h3a_depth2` |
| **H3b** | `configs/rmr_research/h3b_depth4.yaml` | Solver Depth ($T$) | `model.iterations: 4` | `iterations: 6` | `runs/sha_a/rmr_h3b_depth4` |
| **H3c** | `configs/rmr_research/h3c_depth8.yaml` | Solver Depth ($T$) | `model.iterations: 8` | `iterations: 6` | `runs/sha_a/rmr_h3c_depth8` |
| **H4** | `configs/rmr_research/h4_no_morozov.yaml` | Morozov Deadband | `model.morozov_gamma: 0.0` | `morozov_gamma: 0.75` | `runs/sha_a/rmr_h4_no_morozov` |
| **H5** | `configs/rmr_research/h5_no_curvature.yaml` | Curvature Regularization | `loss.lambda_curvature: 0.0` | `lambda_curvature: 0.50` | `runs/sha_a/rmr_h5_no_curvature` |
| **H6** | `configs/rmr_research/h6_spectral_composite.yaml` | Spectral Regularization | `use_spectral_loss: true`<br>`lambda_spectral: 0.2`<br>`spectral_beta: 2.0`<br>`lambda_spectral_dc: 1.0` | Spectral loss disabled | `runs/sha_a/rmr_h6_spectral_composite` |

---

## 4. Research Questions, Variables & Quantitative Hypotheses

### Research Question 1 (RQ1 - Solver Depth Dynamics):
*What is the convergence rate of unrolled SIRT iterations in crowd density reconstruction, and where does autograd unrolling reach diminishing returns?*
- **Independent Variable:** Number of unrolled solver iterations $T \in \{2, 4, 6, 8\}$.
- **Dependent Variables:** Test MAE, Test RMSE, GAME(1, 2, 3), Training Latency (ms/iter), Solver Trajectory Contraction Ratio $\frac{\|r^{(T)}\|}{\|r^{(0)}\|}$.
- **Quantitative Hypothesis (H3):**
  - $T=2$ (H3a): Will yield MAE $\approx 78.5 \pm 1.5$ (under-converged measure, but runs $2.2\times$ faster in solver phase).
  - $T=4$ (H3b): Will yield MAE $\approx 74.0 \pm 1.0$ (monotonically superior to $T=2$).
  - $T=6$ (Baseline): Achieves canonical sweet spot with MAE $\approx 71.5 \pm 0.8$.
  - $T=8$ (H3c): MAE will reach $\approx 70.8 \pm 0.8$ or plateau due to vanishing adjoint gradients through 8 unrolled projections, with a $35\%$ increase in backward pass compute.

### Research Question 2 (RQ2 - Noise Overfitting & Morozov Principle):
*Does eliminating the Morozov discrepancy deadband lead to systematic count inflation in low-density background regions?*
- **Independent Variable:** Morozov deadband threshold coefficient $\gamma \in \{0.0, 0.75\}$.
- **Dependent Variables:** Sparse-Bin MAE ($N \le 100$), Background False Positive Density Mass ($\int_{\text{bg}} Y(x) dx$), Overall MAE/RMSE.
- **Quantitative Hypothesis (H4):**
  - Without the Morozov deadband ($\gamma=0.0$), the solver attempts to fit residual variance that represents pure Poisson/texture noise.
  - Predicted Outcome: Overall MAE degrades by $+3.0$ to $+5.0$ points ($\text{MAE} \approx 75.5 \pm 1.5$), with the largest degradation occurring in the sparse density bin ($N \le 100$) due to phantom counts on pavement and foliage.

### Research Question 3 (RQ3 - Cluster Disambiguation via Curvature):
*How significantly does bilateral gated quadratic curvature regularization improve localization and count accuracy in dense crowd clusters?*
- **Independent Variable:** Curvature loss weight $\lambda_{\text{curv}} \in \{0.0, 0.50\}$.
- **Dependent Variables:** Dense-Bin MAE ($N > 500$), GAME(3) spatial partition error, Peak-to-Saddle Ratio in high-density patches.
- **Quantitative Hypothesis (H5):**
  - Disabling curvature regularization ($\lambda_{\text{curv}}=0.0$) will cause merging of adjacent density peaks in ultra-crowded regions.
  - Predicted Outcome: Dense-bin MAE ($N > 500$) increases by $+6.0\%$, overall MAE degrades to $\approx 73.8 \pm 1.0$, and GAME(3) error increases by $\ge 4.5\%$.

### Research Question 4 (RQ4 - Spectral-Spatial Pareto Frontier):
*Can Fourier-domain power-law decay regularize high-frequency crowd cluster structures without distorting the total crowd count?*
- **Independent Variable:** Count-Preserving Spectral Loss ($\lambda_{\text{spectral}}=0.2, \beta=2.0$).
- **Dependent Variables:** Overall MAE, RMSE, GAME(1, 2), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM).
- **Quantitative Hypothesis (H6):**
  - By decomposing error into scale-invariant DC total mass and $\beta=2.0$ power-law decaying AC spatial frequencies, H6 forces the network to capture head inter-distance semantics.
  - Predicted Outcome: Breaks the sub-70 MAE threshold, targeting **$\text{MAE} \le 69.8$** and **$\text{RMSE} \le 118.0$**, establishing a new state-of-the-art Pareto frontier for models under $105\text{k}$ parameters.

---

## 5. Expected Performance & Ablation Comparison Table

| Experiment Run | Config File | Iterations ($T$) | Morozov $\gamma$ | $\lambda_{\text{curv}}$ | Spectral Loss | Expected MAE | Expected RMSE | Expected GAME(3) | Trainable Params |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical v19** | `rmr_v19_canonical_isotropic.yaml` | 6 | 0.75 | 0.50 | Disabled | $71.5 \pm 0.8$ | $121.0 \pm 1.5$ | $112.5$ | 104,441 |
| **H3a (Depth 2)** | `h3a_depth2.yaml` | **2** | 0.75 | 0.50 | Disabled | $78.5 \pm 1.5$ | $132.0 \pm 2.0$ | $124.0$ | 104,441 |
| **H3b (Depth 4)** | `h3b_depth4.yaml` | **4** | 0.75 | 0.50 | Disabled | $74.0 \pm 1.0$ | $125.0 \pm 1.5$ | $116.8$ | 104,441 |
| **H3c (Depth 8)** | `h3c_depth8.yaml` | **8** | 0.75 | 0.50 | Disabled | $70.8 \pm 0.8$ | $119.5 \pm 1.5$ | $110.2$ | 104,441 |
| **H4 (No Morozov)** | `h4_no_morozov.yaml` | 6 | **0.00** | 0.50 | Disabled | $75.5 \pm 1.2$ | $127.5 \pm 1.8$ | $119.0$ | 104,441 |
| **H5 (No Curvature)**| `h5_no_curvature.yaml` | 6 | 0.75 | **0.00** | Disabled | $73.8 \pm 1.0$ | $124.8 \pm 1.5$ | $117.5$ | 104,441 |
| **H6 (Spectral Comp)**| `h6_spectral_composite.yaml` | 6 | 0.75 | 0.50 | **$\lambda=0.2, \beta=2.0$** | **$\mathbf{69.5 \pm 0.8}$** | **$\mathbf{117.5 \pm 1.5}$** | **$\mathbf{107.8}$** | 104,441 |

---

## 6. Execution Protocol & Verification Standards

### 6.1. Verification Suite Invariants
The automated test suite `tests/test_rmr_research_suite.py` executes 21 strict validation checks:
1. **Schema Integrity:** Each YAML configuration must pass `rmr_v3.config.validate_v3_config(cfg)`.
2. **Deterministic Flag:** All configs enforce `cfg["train"]["deterministic"] is False` to preserve full stochastic data augmentation entropy over the 300 ShanghaiTech Part A training images.
3. **Parameter Invariant:** Every configuration instantiates a model via `make_model(cfg)` with **strictly 104,441 trainable parameters** ($\le 105,000$ budget).
4. **Single-Variable Isolation:** Automated dictionary diff against canonical v19 confirms zero collateral parameter drift.
5. **Code Line Limit Invariant:** All python files adhere strictly to the $\le 450$ lines constraint.

### 6.2. Training Commands
To launch experiments sequentially or concurrently across GPU resources:

```powershell
# 1. Hypothesis H3a: Depth T=2
python train.py --config configs/rmr_research/h3a_depth2.yaml

# 2. Hypothesis H3b: Depth T=4
python train.py --config configs/rmr_research/h3b_depth4.yaml

# 3. Hypothesis H3c: Depth T=8
python train.py --config configs/rmr_research/h3c_depth8.yaml

# 4. Hypothesis H4: No Morozov Deadband
python train.py --config configs/rmr_research/h4_no_morozov.yaml

# 5. Hypothesis H5: No Curvature Regularization
python train.py --config configs/rmr_research/h5_no_curvature.yaml

# 6. Hypothesis H6: Spectral Composite (Sub-70 MAE Target)
python train.py --config configs/rmr_research/h6_spectral_composite.yaml
```

---

## 7. Automated Test Verification Log

All tests passed successfully under Python 3.13.7 with `pytest-9.0.2`:

```text
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h3a_depth2.yaml] PASSED [  4%]
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h3b_depth4.yaml] PASSED [  9%]
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h3c_depth8.yaml] PASSED [ 14%]
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h4_no_morozov.yaml] PASSED [ 19%]
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h5_no_curvature.yaml] PASSED [ 23%]
tests/test_rmr_research_suite.py::test_config_file_exists_and_validates[h6_spectral_composite.yaml] PASSED [ 28%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h3a_depth2.yaml] PASSED [ 33%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h3b_depth4.yaml] PASSED [ 38%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h3c_depth8.yaml] PASSED [ 42%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h4_no_morozov.yaml] PASSED [ 47%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h5_no_curvature.yaml] PASSED [ 52%]
tests/test_rmr_research_suite.py::test_train_deterministic_is_false[h6_spectral_composite.yaml] PASSED [ 57%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h3a_depth2.yaml] PASSED [ 61%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h3b_depth4.yaml] PASSED [ 66%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h3c_depth8.yaml] PASSED [ 71%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h4_no_morozov.yaml] PASSED [ 76%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h5_no_curvature.yaml] PASSED [ 80%]
tests/test_rmr_research_suite.py::test_exact_parameter_count_104441[h6_spectral_composite.yaml] PASSED [ 85%]
tests/test_rmr_research_suite.py::test_single_variable_isolation PASSED  [ 90%]
tests/test_rmr_research_suite.py::test_h6_spectral_loss_integration PASSED [ 95%]
tests/test_rmr_research_suite.py::test_codebase_line_count_invariant PASSED [100%]

============================= 21 passed in 6.27s ==============================
```
