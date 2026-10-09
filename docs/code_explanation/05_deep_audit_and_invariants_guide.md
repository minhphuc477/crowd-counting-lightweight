# Chapter 5: Deep Architectural Audit & Mathematical Invariants Guide

This document details the automated diagnostic methodology, mathematical invariants, and systematic testing framework implemented in [`tests/core/test_systematic_deep_audit.py`](file:///f:/lightweightcrcn/tests/core/test_systematic_deep_audit.py) and [`.agents/skills/ml-model-deep-audit/SKILL.md`](file:///f:/lightweightcrcn/.agents/skills/ml-model-deep-audit/SKILL.md).

---

## 1. Why ML Code Review Differs from Traditional SE

Traditional software engineering tests for crashes, exceptions, and logical branches.
In Machine Learning and Deep Learning, **bugs are almost always silent**:
* The loss function decreases smoothly, but the model has learned an unphysical shortcut.
* A subtle tensor broadcast `[B, 1] - [M]` accidentally expands into `[B, M, M]`, silently passing autograd while computing invalid gradients.
* Samples within a batch influence each other's predictions due to cross-sample reductions or BatchNorm leakage.
* High-frequency spatial detail is destroyed due to half-pixel offsets.

To safeguard against these failure modes, RMR-v3 enforces a **Dynamic Invariant Test Suite** executed before every training deployment.

---

## 2. The 7 Core Architectural Invariants

### 2.1. Invariant 1: P0 Batch Sample Independence
A model $f(x)$ must satisfy the independence property: the prediction and autograd gradient for an image $x_i$ evaluated as part of a batch $[x_i, x_j]$ must match its evaluation in isolation:
$$f(x_i) \equiv \left[ f([x_i, x_j]) \right]_0$$
$$\nabla_{x_i} \mathcal{L}(f(x_i)) \equiv \left[ \nabla_{[x_i, x_j]} \mathcal{L}(f([x_i, x_j])) \right]_0$$
* **Tolerance:** Absolute difference $< 1 \times 10^{-5}$.
* **Failure Significance:** Catches BatchNorm tracking leakage, improper cross-batch normalization, or global reduction bugs.
* **Verification Function:** `test_batch_sample_independence_invariant`.

---

### 2.2. Invariant 2: L1 Mass Conservation Under Horizontal Flip
Crowd headcount is an intrinsic physical scalar invariant to horizontal reflection:
$$\int_{\Omega} \hat{y}(u, v) \, du \, dv \equiv \int_{\Omega} \hat{y}(\text{flip}(u, v)) \, du \, dv$$
* **Tolerance:** Relative count discrepancy $< 5.0\%$.
* **Failure Significance:** Catches asymmetric padding, asymmetric kernel biases, or coordinate grid orientation errors.
* **Verification Function:** `test_l1_count_mass_conservation_under_flip`.

---

### 2.3. Invariant 3: Dual-Lattice Push-Forward Exact Conservation
When projecting between the Stride 2 fine lattice ($256 \times 256$) and Stride 4 coarse solver lattice ($128 \times 128$), total discrete mass must be strictly conserved:
$$\sum_{u, v} y_{\text{fine}}(u, v) \equiv \sum_{i, j} y_{\text{coarse}}(i, j)$$
* **Tolerance:** Relative error $< 1 \times 10^{-6}$.
* **Failure Significance:** Prevents false mass inflation or loss when switching between solver and subpixel heads.
* **Verification Function:** `test_dual_lattice_pushforward_exact_conservation`.

---

### 2.4. Invariant 4: Autograd Graph Continuity on Prime Spatial Shapes
Image dimensions in real-world benchmarks are arbitrary. Convolutional and pooling operations can encounter odd or prime dimensions (e.g. $251 \times 257$):
* Forward and backward autograd must succeed with zero NaN, Inf, or dimension mismatch exceptions.
* In-place operations (`+=`, `.mul_()`) must never corrupt graph backpropagation.
* **Verification Function:** `test_autograd_graph_and_numerical_safety_under_prime_shapes`.

---

### 2.5. Invariant 5: Lipschitz Residual Energy Dissipation
The unrolled SIRT solver in RMR-v3 implements a relaxed Picard iteration:
$$y^{(t+1)} = \text{proj}_+ \left( y^{(t)} + \omega \Delta y^{(t)} \right)$$
Under contraction mapping guarantees, the weighted residual energy $E(y^{(t)}) = \| \mathcal{A} y^{(t)} - b \|_W^2$ must dissipate monotonically over solver iterations:
$$E(y^{(t+1)}) \le E(y^{(t)}), \quad \forall t \in \{1, \dots, T-1\}$$
* **Verification Function:** `test_unrolled_solver_lipschitz_residual_dissipation`.

---

### 2.6. Invariant 6: Bottleneck SVD Effective Rank ($\text{erank}$)
Feature maps in ultra-lightweight backbones ($< 105\text{k}$ parameters) can suffer from *Representation Collapse*, where 32 feature channels collapse into an effectively 1- or 2-dimensional subspace.

Effective Rank measures the Shannon entropy of the normalized singular value spectrum:
$$p_k = \frac{\sigma_k}{\sum_{j=1}^K \sigma_j}, \quad H(p) = -\sum_{k=1}^K p_k \ln(p_k)$$
$$\text{erank}(F) = \exp(H(p))$$
* **Threshold:** $\frac{\text{erank}(P_4)}{C_{\text{features}}} \ge 15\%$.
* **Measured Result:** On MobileNetV4-Conv-Small-050, the normalized effective rank is $\approx 19.4\%$, confirming rich feature diversity without rank collapse.
* **Verification Function:** `test_bottleneck_feature_effective_rank`.

---

### 2.7. Invariant 7: Exact Half-Pixel Coordinate Symmetry
The center of cell $(r, c)$ in continuous pixel space must be symmetric under reflection:
$$x_{\text{center}}(c) = (c + 0.5) \cdot \text{stride}$$
$$x_{\text{center}}(W - 1 - c) + x_{\text{center}}(c) \equiv W \cdot \text{stride}$$
* **Failure Significance:** An offset of $0.5\text{ px}$ creates systematic directional drift during backpropagation.
* **Verification Function:** `test_half_pixel_center_coordinate_symmetry`.

---

## 3. Codebase Monolith Prevention Rules

The test [`TestCodebaseMonolithPrevention`](file:///f:/lightweightcrcn/tests/rmr_v3/test_rmr_v30_m2_parity.py#L245) enforces two strict invariants:
1. **File Line Ceiling:** Every Python source file in `rmr_core/` and `rmr_v3/` must have $\le 450$ lines.
2. **File Count Boundary:** Total audited Python source files must remain within $[58, 85]$ files (currently 82 files).

### How to Maintain This Invariant
* When a module approaches 400 lines, extract helper factories (e.g. [`rmr_v3/model/builder.py`](file:///f:/lightweightcrcn/rmr_v3/model/builder.py)).
* Keep schemas, validators, and operators in separate files.
* Never inline large dictionary lookups or duplicate parameter validation.

---

## 4. Running the Verification Suite

To run all deep audit and architectural invariant tests locally before submitting changes or launching training:

```bash
# Run the 7 systematic deep audit invariant tests
pytest tests/core/test_systematic_deep_audit.py -v

# Run the full baseline parity and monolith prevention suite
pytest tests/rmr_v3/test_rmr_v30_m2_parity.py -v

# Run both suites together (< 10 seconds total)
pytest tests/core/test_systematic_deep_audit.py tests/rmr_v3/test_rmr_v30_m2_parity.py -v
```
All 40 tests must pass with zero failures and zero warnings.
