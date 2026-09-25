# RMR-v34: Dynamic Image-Adaptive Geometry (DiAG) & Discrete Sparse Measure Protection (DSMP)
## Authoritative Architecture Specification & Empirical Foundation Treatise

---

### Executive Summary & Scientific Charter
This document serves as the permanent, authoritative architectural and mathematical specification for **RMR-v34** (Radon Measure Recovery v34). It establishes:
1. The **exact mathematical framework** of continuous-discrete Radon measure estimation for dense crowd counting under extreme perspective distortions.
2. The **forensic autopsy** of all prior iterations—specifically explaining why the earlier H11 baseline achieved **71.20 MAE (TTA)** while the v33 PARK iteration regressed to **80.52 MAE**.
3. The complete design of **Dynamic Image-Adaptive Geometry (DiAG)**, which eliminates static spatial coordinates and elevation bands in favor of translation-equivariant, feature-driven scale routing.
4. The formulation of **Discrete Sparse Measure Protection (DSMP)**, unifying count-invariant cell supervision, top-$K$ hard background mining, hurdle occupancy masking, and minimax concave penalty (MCP) proximal thresholding.
5. Strict adherence to scientific constraints: parameter budget $\le 105,000$ (exact: **104,701** parameters with Vertical Differential Pooling DCAP; **104,573** without VDP), zero knowledge distillation, modular files $\le 450$ lines, and zero static spatial coordinate grids (`torch.linspace`).

---

## 1. Mathematical Formulation: Continuous-Discrete Radon Measure Recovery

### 1.1 The Measure-Theoretic Formulation
Crowd counting with point annotations represents an ill-posed inverse problem defined on a bounded spatial domain $\Omega \subset \mathbb{R}^2$. The true population distribution is a discrete Radon measure:
$$\mu = \sum_{i=1}^{N} c_i \delta_{x_i}, \quad x_i \in \Omega, \ c_i > 0$$
where $N$ is the true total count and $\delta_{x_i}$ is the Dirac measure centered at point $x_i$.

Standard crowd counting approaches approximate $\mu$ by convolving discrete points with a 2D Gaussian kernel $G_\sigma(x)$ to construct a surrogate continuous density map $D(x) = (\mu * G_\sigma)(x)$. However, Gaussian blurring introduces severe artifacts:
- In dense regions ($\|x_i - x_j\| < 2\sigma$), individual Dirac masses merge into smooth blobs, destroying high-frequency topological separation.
- In sparse regions, Gaussian tails disperse probability mass over non-crowd pixels, exacerbating false-positive accumulation.

### 1.2 The Forward Measurement Operator and Dual Adjoint
RMR formulates crowd estimation as the inversion of an observation operator $\mathcal{A}: \mathcal{M}(\Omega) \to \mathbb{R}^M$. For a family of multiscale measurement regions $\mathcal{R} = \{R_m\}_{m=1}^M$, the forward operator evaluates regional integrals:
$$y_m = (\mathcal{A} \mu)_m = \int_{R_m} d\mu(x) = \sum_{x_i \in R_m} c_i$$

The adjoint operator $\mathcal{A}^*: \mathbb{R}^M \to C(\Omega)$ maps regional measurements back to continuous spatial density:
$$(\mathcal{A}^* y)(x) = \sum_{m=1}^M y_m \cdot \mathbf{1}_{R_m}(x)$$

In RMR-v34, $\mathcal{A}$ comprises a multiscale universal dictionary of pooling regions $\mathcal{R}$ at scales $s \in \{32, 64, 128\}$ px with overlapping spatial strides:
- **Small regions ($32 \times 32$ px, stride 16)**: $N_{32} = 961$ boxes (for a $512 \times 512$ canvas).
- **Medium regions ($64 \times 64$ px, stride 32)**: $N_{64} = 225$ boxes.
- **Large regions ($128 \times 128$ px, stride 64)**: $N_{128} = 49$ boxes.
- **Total observation count**: $M = 961 + 225 + 49 = 1,235$ constraints everywhere across $\Omega$.

---

## 2. Forensic Autopsy: Why H11 Succeeded (71.20 MAE) and v33 Failed (80.52 MAE)

To eliminate recurring architectural regressions, Table 1 details the exact structural and mathematical mechanisms separating the peak H11 performance from the degraded v33 PARK iteration.

### Table 1: Forensic Architectural and Loss Comparison
| Component | Peak Baseline (H11: 71.20 MAE) | Degraded Iteration (v33 PARK: 80.52 MAE) | RMR-v34 DiAG + DSMP (Target) |
|---|---|---|---|
| **Measurement Dictionary $\mathcal{R}$** | **Universal Multiscale**: All 3 scales everywhere ($M = 1,235$ regions). | **Hard Altitude Partition**: Top 40% small, mid 20% medium, bottom 40% large ($M = 502$ regions). | **Universal Multiscale**: All 3 scales active across entire canvas ($M = 1,235$). |
| **Observation Constraints** | $1,235$ regional integrals ($100\%$). | $502$ regional integrals (**$-59.4\%$ loss of constraints**). | $1,235$ regional integrals ($100\%$). |
| **Foreground Small Boxes** | Active (32px boxes in foreground capture dense clumps). | **Zero 32px boxes in bottom 40%**. Dense clumps forced into 128px boxes. | Active everywhere; routing dynamically weights scales. |
| **Routing Mechanism** | Soft feature-conditioned scale mixture. | Rigid coordinate-band index clamping ($K=2$ head vs 3 bands). | **DiAG**: Continuous semantic tilt + scale bias + carrier gating. |
| **Hurdle Gating** | Hurdle occupancy head zeroes background prior to solver. | Hurdle present but bypassed by unweighted solver residual. | **DSMP Hurdle**: Strict gating $b_{\text{solver}} = \pi_R \cdot \mu_R$. |
| **Cell Supervision** | **CI-Cell v2** ($\alpha=2.0, \beta=1.0$ count-invariant weighting). | Standard L1 / Flat MSE (dense clusters penalized weakly). | **CI-Cell v2**: Preserved with high-density gradient amplification. |
| **Background Mining** | **Top-$K$ Hard BG Loss** ($\lambda=0.15$, 5% hardest false positives). | Removed / inactive. | **Top-$K$ Hard BG Mining**: Active ($\lambda=0.15$). |
| **Iterative Inversion** | **Resonant Adjoint** with momentum $\beta_{\text{res}} = 0.25$. | Flat unrolled Landweber without momentum damping. | **Resonant Adjoint Landweber**: Preserved ($T=2, \beta_{\text{res}}=0.25$). |
| **Thresholding** | **Proximal MCP** ($\tau=0.015, \mu=3.0$). | Hard ReLU / Soft shrinkage. | **Proximal MCP Firm Thresholding**: Preserved. |
| **Coordinate Geometry** | Zero coordinate grids; pure feature representations. | Attempted $(u, v)$ elevation grids (`torch.linspace`). | **Zero Coordinate Grids**: 100% translation equivariant. |

### 2.1 The Two Catastrophic Flaws of v33 PARK
1. **Geometric Underfitting via Constraint Decimation**:
   By forcing images into hard vertical elevation bands (Band 0: small, Band 1: medium, Band 2: large), v33 eliminated 733 fine-grained observation boxes. In ShanghaiTech Part A, images frequently feature high-angle overhead or steep perspective cameras where hundreds of small heads appear in the bottom half of the frame. Because v33 provided zero 32px boxes in the bottom 40%, the solver was mathematically incapable of resolving dense local clusters, driving Dense MAE from $112.92$ up to $145.40$ ($+32.48$ error explosion).
2. **Dimension Mismatch & Routing Reversal**:
   `PARKRoutingHead` emitted 2 scale logits, while the region partition contained 3 altitude bands. Band 2 (all 45 foreground 128px boxes) was hard-clamped into Band 1 weights. When routing was ablated in v33 (`no_routing`), MAE improved from $80.52$ to $79.16$—proving the routing head was actively degrading representations!

---

## 3. Dynamic Image-Adaptive Geometry (DiAG)

### 3.1 Principles of Pure Feature-Conditioned Routing
DiAG completely rejects static coordinate grids (`torch.linspace`), normalized spatial coordinates $(u, v)$, and rigid horizontal bands. In real-world surveillance and crowd scenes, camera angles vary from extreme oblique perspectives to pure overhead nadir views. A static spatial coordinate prior breaks under random cropping during training and causes severe out-of-distribution shifts at test time.

Instead, DiAG computes camera perspective and scale distributions dynamically from image features at two distinct semantic levels:
1. **Global Scene Perspective**: Extracted from deep feature representations ($P_{16}$).
2. **Local Scale Carrier**: Modulated across high-resolution spatial feature maps ($P_4$).

```
Image X (B, 3, H, W)
  │
  ├──> Backbone (Stem + Stages) ──> P4 (B, 32, H/4, W/4), P8, P16 (B, 32, H/16, W/16)
  │                                   │                      │
  │                                   │                      ├──> Global Avg Pool (GAP)
  │                                   │                      └──> DCAP Linear(32, 4)
  │                                   │                             ├──> Scene Tilt ∈ [0, 1]
  │                                   │                             └──> Scale Bias δ_scale ∈ R^3
  │                                   │                                     │
  │                                   ▼                                     ▼
  │                           DiAGScaleRoutingHead ◄────────────────────────┘
  │                             (DW-Conv3x3 + GN + SiLU + PW-Conv1x1)
  │                             Logits = Logits * (0.5 + Scene_Tilt) + δ_scale
  │                             Softmax over K=3 scales
  │                                   │
  │                                   ▼
  │                           Scale Weights π(x, y) ∈ Δ² (B, 3, H/4, W/4)
```

### 3.2 Dynamic Camera Angle Predictor (DCAP)
DCAP operates on the deepest feature stage $P_{16} \in \mathbb{R}^{B \times 32 \times \frac{H}{16} \times \frac{W}{16}}$:
$$\bar{z} = \frac{16^2}{H \cdot W} \sum_{h=1}^{H/16} \sum_{w=1}^{W/16} P_{16}[:, :, h, w] \in \mathbb{R}^{B \times 32}$$
$$\phi = W_{\text{dcap}} \bar{z} + b_{\text{dcap}}, \quad W_{\text{dcap}} \in \mathbb{R}^{4 \times 32}, \ b_{\text{dcap}} \in \mathbb{R}^4$$
The 4-dimensional projection partitions into:
- **Perspective Tilt Factor**: $\text{scene\_tilt} = \sigma(\phi_0) \in (0, 1)$.
- **Global Scale Prior**: $\delta_{\text{scale}} = \phi_{1:3} \in \mathbb{R}^{B \times 3 \times 1 \times 1}$.

### 3.3 DiAG Routing Head Formulation
The local carrier map $P_4 \in \mathbb{R}^{B \times 32 \times \frac{H}{4} \times \frac{W}{4}}$ is processed by depthwise separable convolution:
$$Z_{\text{local}} = \text{PWConv}\left(\text{SiLU}\left(\text{GN}\left(\text{DWConv}_{3\times 3}(P_4)\right)\right)\right) \in \mathbb{R}^{B \times 3 \times \frac{H}{4} \times \frac{W}{4}}$$

The raw logits are modulated by the global scene perspective:
$$\mathcal{S}(x, y) = Z_{\text{local}}(x, y) \cdot (0.5 + \text{scene\_tilt}) + \delta_{\text{scale}}$$
$$\pi_k(x, y) = \frac{\exp(\mathcal{S}_k(x, y))}{\sum_{j=1}^3 \exp(\mathcal{S}_j(x, y))}, \quad k \in \{1, 2, 3\}$$

#### Step 0 Initialization Identity
At step 0 of training, weights are initialized such that:
$$W_{\text{dcap}} = 0, \quad b_{\text{dcap}} = [0, 0, 0, 0]^T$$
$$Z_{\text{local}} = 0, \quad \text{scene\_tilt} = \sigma(0) = 0.5 \implies (0.5 + 0.5) = 1.0$$
$$\mathcal{S}_k(x, y) = 0 \implies \pi_k(x, y) = \frac{1}{3} \quad \forall k \in \{1, 2, 3\}$$
This guarantees exact uniform identity parity at step 0 without transient training instability. During training, steep perspective angles ($\text{scene\_tilt} > 0.5$) dynamically sharpen local scale transitions, while flat nadir views ($\text{scene\_tilt} < 0.5$) smooth them.

---

## 4. Discrete Sparse Measure Protection (DSMP)

DSMP guarantees numerical and mathematical integrity across the continuous-discrete boundary through four complementary mechanisms:

### 4.1 Hurdle Occupancy Gating
To eliminate background noise from contaminating the inverse solver, the Hurdle Head predicts binary regional occupancy $\pi_R \in [0, 1]$ alongside continuous density intensity $\mu_R \ge 0$:
$$b_{\text{solver}} = \pi_R \odot \mu_R$$
When a measurement region contains zero people ($\pi_R < 0.1$), its effective measurement in the Landweber residual is suppressed to zero, preventing false-positive gradient updates.

### 4.2 Count-Invariant Two-Stream Cell Loss v2 (CI-Cell v2)
Standard L1/L2 cell losses under-supervise dense clumps because the relative error on large counts ($> 50$) generates gradient magnitudes identical to isolated errors in sparse backgrounds. CI-Cell v2 balances supervision across density regimes:
$$\mathcal{L}_{\text{cell}} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{|y_c - \hat{y}_c|}{(y_c + \epsilon)^\alpha + \beta}$$
with $\alpha = 2.0$, $\beta = 1.0$, and $\epsilon = 1.0$. This prevents gradient saturation on dense clusters while maintaining sharp boundaries.

### 4.3 Top-$K$ Hard Background Mining Loss
Background false positives in crowd counting typically concentrate in a tiny fraction of ambiguous regions (tree leaves, building facades, textures). Standard average losses dilute these hard errors:
$$\mathcal{L}_{\text{hard\_bg}} = \frac{1}{K_{\text{bg}}} \sum_{i \in \text{Top-}K(\mathcal{R}_{\text{empty}})} \hat{y}_i$$
where $\mathcal{R}_{\text{empty}} = \{m \in \mathcal{R} \mid y_m = 0\}$ and $K_{\text{bg}} = \lfloor 0.05 \cdot |\mathcal{R}_{\text{empty}}| \rfloor$. The loss applies weight $\lambda_{\text{hard\_bg}} = 0.15$.

### 4.4 Proximal Firm Thresholding (MCP)
During unrolled SIRT/Landweber iterations, intermediate density estimates $\rho^{(t)}$ are regularized using the Minimax Concave Penalty (MCP) proximal operator:
$$\text{prox}_{\tau, \mu}(u) = \begin{cases}
0, & |u| \le \tau \\
\frac{\text{sign}(u)(|u| - \tau)}{1 - 1/\mu}, & \tau < |u| \le \mu \tau \\
u, & |u| > \mu \tau
\end{cases}$$
with threshold $\tau = 0.015$ and concavity parameter $\mu = 3.0$. Unlike $\ell_1$ soft thresholding (which introduces permanent bias on high counts), MCP behaves as soft thresholding for small noise while transitioning to unbiased identity for true dense peaks.

---

## 5. Composite Loss Formulation & Optimization

The complete objective function optimized during training is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flat}} + \lambda_{\text{otm}} \mathcal{L}_{\text{otm}} + \lambda_{\text{hard\_bg}} \mathcal{L}_{\text{hard\_bg}} + \lambda_{\text{cell}} \mathcal{L}_{\text{cell}} + \lambda_{\text{scale}} \mathcal{L}_{\text{scale}}$$

Where:
- $\mathcal{L}_{\text{flat}}$: Flat Dirichlet-Multinomial (DM-Count) count and optimal transport density loss on $P_4$ ($\lambda_{\text{flat}} = 1.0$).
- $\mathcal{L}_{\text{otm}}$: Optimal Transport Matching loss between predicted Dirac measures and ground truth points ($\lambda_{\text{otm}} = 0.05$).
- $\mathcal{L}_{\text{hard\_bg}}$: Top-$K$ hard background mining loss on empty measurement regions ($\lambda_{\text{hard\_bg}} = 0.15$).
- $\mathcal{L}_{\text{cell}}$: CI-Cell v2 count-invariant multiscale cell supervision ($\lambda_{\text{cell}} = 0.5$).
- $\mathcal{L}_{\text{scale}}$: DiAG scale alignment cross-entropy loss against ground truth scale distribution ($\lambda_{\text{scale}} = 0.1$).

---

## 6. Model Parameter Audit & Architectural Invariants

### 6.1 Strict Parameter Accounting ($\le 105,000$ Target)
Every parameter in RMR-v34 is tracked and budgeted as shown in Table 2:

### Table 2: Complete Module Parameter Inventory
| Component | Module Name | Trainable Parameters | Description |
|---|---|---|---|
| **Backbone Stem & Stages** | `backbone` | 87,568 | Inverted residual MobileNetV4 / ConvNeXt-Femto blocks |
| **ASPP-Lite Neck** | `neck` | 10,784 | Dilated multiscale receptive field fusion at stride 4 |
| **Fine Density Head** | `fine_head` | 1,475 | Continuous density anchor generator ($y_0$) with curvature |
| **Region Head (DSMP)** | `region_head` | 4,131 | Probabilistic regional evidence and Hurdle occupancy head |
| **DCAP Perspective (VDP)** | `dcap` | 260 | Vertical Differential Pooling + GAP + Linear(64, 4) |
| **DiAG Scale Router** | `scale_router` | 483 | Depthwise-separable $3\times 3$ Conv + GroupNorm + $1\times 1$ Conv |
| **Iterative Inversion Solver** | Algorithmic (`rmr_core`) | 0 (Shared / Non-Param) | Unrolled Landweber solver with resonant momentum |
| **Total Trainable** | **Full Canonical Model** | **104,701** | **Budget: $\le 105,000$ (Margin: 299 params)** |

### 6.2 Code Quality & Engineering Invariants
- **File Line Length Ceiling**: Every Python source file in `rmr_core/` and `rmr_v3/` must strictly not exceed 450 lines. Current maximum line count across all 58 source files is **446 lines** (`rmr_v3/model/architecture.py`).
- **Translation Equivariance**: Zero coordinate grids (`torch.linspace`, $(u, v)$ meshes) exist in any model layer or operator.
- **Standalone Topology**: Zero teacher models, zero distillation losses, zero external pretrained weights. The model trains 100% end-to-end from scratch.
- **Active Gradient Propagation**: Every single submodule (including DCAP tilt projection, DiAG carrier modulation, and hurdle gating) receives non-zero backpropagation gradients.

---

## 7. Multi-Dataset Evaluation & Systematic Ablation Suite

### 7.1 Evaluation Benchmarks
Evaluation is performed uniformly across standard public benchmarks using official metrics (MAE, RMSE, and Game(4)):
1. **ShanghaiTech Part A (SHA)**:
   - High-density, extreme perspective variations ($N = 300$ train, $182$ test).
   - Validates small-scale resolution and dense clump de-aggregation.
2. **ShanghaiTech Part B (SHB)**:
   - Sparse, wide-angle outdoor surveillance ($N = 400$ train, $316$ test).
   - Validates false-positive suppression via DSMP top-$K$ hard background mining.

### 7.2 The 17-Experiment RMR-v34 Ablation Matrix
To guarantee publication-grade empirical rigor, the RMR-v34 suite isolates each constituent mechanism via single-variable hypothesis testing:

| Config Key | Category | Trainable Params | Targeted Hypothesis / Variable |
|---|---|---|---|
| `rmr_v34_diag_canonical.yaml` | Baseline | 104,701 | Canonical reference (DiAG with VDP DCAP, DSMP, $T=6$, SNR Weighting, MCP Firm). |
| `rmr_v34_shb_canonical.yaml` | Cross-Dataset | 104,701 | ShanghaiTech Part B benchmark evaluation (316 test images). |
| `rmr_v34_abl_no_vdp.yaml` | DCAP Upgrade | 104,573 | Ablate Vertical Differential Pooling (`use_vertical_gradient_dcap: false`, 2D GAP). |
| `rmr_v34_abl_scale_prior.yaml` | Prior Coupling | 104,707 | Test scale-conditioned FineHead prior coupling (`scale_conditioned_prior: true`). |
| `rmr_v34_abl_no_diag.yaml` | DiAG Routing | 103,958 | Ablate DiAG routing (`use_diag: false`, uniform isotropic multiscale weights). |
| `rmr_v34_abl_no_dcap_tilt.yaml` | DiAG Routing | 104,701 | Ablate scene tilt contrast scaling (`use_dcap_tilt: false`). |
| `rmr_v34_abl_no_scale_align.yaml` | DiAG Routing | 104,701 | Ablate scale alignment loss (`lambda_scale_align: 0.0`). |
| `rmr_v34_abl_no_hurdle.yaml` | DSMP Protection | 104,652 | Ablate hurdle occupancy gating (`hurdle_head: false, lambda_hurdle: 0.0`). |
| `rmr_v34_abl_no_hard_bg.yaml` | DSMP Protection | 104,701 | Ablate top-K hard background mining (`lambda_hard_bg: 0.0`). |
| `rmr_v34_abl_no_ci_cell.yaml` | DSMP Protection | 104,701 | Ablate CI-Cell v2 count-invariance (`cell_loss_mode: balanced`). |
| `rmr_v34_abl_no_proximal.yaml` | DSMP Protection | 104,701 | Ablate proximal thresholding (`proximal_mode: none`). |
| `rmr_v34_abl_soft_proximal.yaml` | DSMP Protection | 104,701 | Test soft thresholding vs firm MCP (`proximal_mode: soft`). |
| `rmr_v34_abl_no_solver.yaml` | Inverse Solver | 104,701 | Ablate unrolled solver ($T=0$, `enable_solver: false`, direct anchor $y_0$). |
| `rmr_v34_abl_solver_t2.yaml` | Inverse Solver | 104,701 | Fast solver contraction depth ($T=2$ iterations vs canonical $T=6$). |
| `rmr_v34_abl_solver_t8.yaml` | Inverse Solver | 104,701 | Deep solver contraction depth ($T=8$ iterations vs canonical $T=6$). |
| `rmr_v34_abl_asym_morozov.yaml` | Inverse Solver | 104,701 | Asymmetric Morozov discrepancy principle (`asymmetric_morozov: true`). |
| `rmr_v34_abl_no_resonant.yaml` | Inverse Solver | 104,701 | Ablate carrier Laplacian momentum (`resonant_adjoint: false`). |
| `rmr_v34_abl_no_curvature.yaml` | Inverse Solver | 104,700 | Ablate density curvature regularization (`lambda_curvature: 0.0`). |
| `rmr_v34_abl_uniform_reliability.yaml` | Inverse Solver | 104,701 | Ablate SNR reliability weighting (`uniform_reliability: true`). |
| `rmr_v34_seed123.yaml` | Multi-Seed | 104,701 | Statistical variance verification (`seed: 123`). |
| `rmr_v34_seed456.yaml` | Multi-Seed | 104,701 | Statistical variance verification (`seed: 456`). |

---

## 8. Conclusion & Canonical Directive
RMR-v34 establishes a mathematically sound, empirical framework for crowd counting as continuous-discrete Radon measure recovery. By replacing the brittle, flawed static altitude partitions of v33 with dynamic, feature-driven DiAG scale routing and DSMP protection, RMR-v34 restores full observation resolution while strictly respecting the 105,000 parameter budget.
