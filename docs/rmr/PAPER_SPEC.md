# RMR-Count: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

> **Target Venue:** IEEE / CVPR 2026 (Computer Vision and Pattern Recognition)  
> **Topic:** Ultra-Lightweight Visual Crowd Analysis, Operator-Guided Optimization Layers, Discrete Geometric Measure Theory.  
> **Scope:** Pure visual crowd counting and spatial density estimation (< 105k parameters). This is an explicit counting architecture; point localization, detection bounding boxes, and Hungarian matching are out of scope.

---

## 1. Abstract & Motivation

In ultra-lightweight crowd counting (< 105k parameters), models must maintain local spatial resolution while preserving global count coherence under strict memory and computational constraints. Heavy crowd counters rely on deep transformer backbones, dense self-attention, or large multi-scale dilated convolutions to integrate long-range context. However, these mechanisms exceed the parameter and compute budget of mobile-edge carriers.

We investigate a fundamental research question:
$$\boxed{\textbf{Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?}}$$

**RMR-Count** (Regional Measure Reconciliation) answers this question through a decoupled, operator-guided architecture:
1. A **Pretrained MobileNetV4 Carrier & Additive FPN Neck** extracts multi-scale representations $(P_4, P_8, P_{16})$ with dilated convolutions.
2. A **Fine Measure Head** estimates initial spatial cell densities $Y_0 \in \mathbb{R}_+^G$ at output stride $s=4$, supervised via Flat Dirichlet-Multinomial-16 (Flat-DM16) and count losses.
3. A **Scale-Matched Regional Evidence Head** independently predicts aggregated count mass $b \in \mathbb{R}_+^M$ across multi-scale bounding regions $\mathcal{R}$ routed by physical window size.
4. An **Operator-Guided Unrolled Reconciliation Layer (Projected SIRT)** iteratively reconciles local densities against regional count constraints directly in measure space using the **exact geometric adjoint operator** $A^\top$ and nonnegativity projection $\Pi_+$.

Crucially, we prove that under the area-normalized and coverage-preconditioned transfer operator $H = D_c^{-1} A^\top D_a^{-1} A$, the system satisfies the **Adjoint Scale Invariance Theorem**: $H \mathbf{1}_G = \mathbf{1}_G$. This guarantees transfer invariance: a uniform regional rate discrepancy induces an identically uniform spatial correction field across covered cells, preserving constant fields across arbitrary multi-scale partitions without scale-dependent gradient explosion or boundary artifacts.

---

## 2. Mathematical Formulation

### 2.1 Spatial Discretization & Ground-Truth Cell Representation
Let an input image $I \in \mathbb{R}^{3 \times H \times W}$ contain $N$ annotated head points $\mathcal{P} = \{(x_n, y_n)\}_{n=1}^N$.
Under output stride $s=4$, the spatial counting lattice has dimensions $H_o = \lceil H/s \rceil, W_o = \lceil W/s \rceil$ with total grid cells $G = H_o \cdot W_o$.

Points strictly outside the image support $(x < 0, x \ge W, y < 0, y \ge H)$ are filtered out. The canonical ground-truth count per discrete cell is:
$$Y_{ij}^* = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left(\left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j\right),$$
with cell indices clamped to the valid lattice bounds $[0, H_o - 1] \times [0, W_o - 1]$, ensuring exact count conservation $\sum_{i,j} Y_{ij}^* = |\mathcal{P}_{\text{valid}}|$.

### 2.2 Regional Projection Operator $A$ and Adjoint $A^\top$
We define a multi-scale regional dictionary $\mathcal{R} = \{R_m\}_{m=1}^M$ spanning window sizes $K \in \{32, 64, 128\}$ pixels (grid dimensions $k = K/s \in \{8, 16, 32\}$ cells) with stride overlap $\sigma = 0.5$.

- **Forward Regional Projection Matrix** $A \in \{0, 1\}^{M \times G}$:
  $$(A Y)_m = \sum_{g \in R_m} Y_g = q_m.$$
- **Adjoint Back-Projection Matrix** $A^\top \in \{0, 1\}^{G \times M}$:
  $$(A^\top r)_g = \sum_{m: g \in R_m} r_m.$$

### 2.3 The Adjoint Scale Invariance Theorem ($H \mathbf{1} = \mathbf{1}$)
Let $D_a \in \mathbb{R}^{M \times M}$ be the diagonal matrix of regional areas: $(D_a)_{mm} = |R_m|$.  
Let $D_c \in \mathbb{R}^{G \times G}$ be the diagonal matrix of cell coverage counts: $(D_c)_{gg} = \sum_{m=1}^M \mathbf{1}(g \in R_m) = (A^\top \mathbf{1}_M)_g$.

We define the normalized regional transfer operator:
$$H = D_c^{-1} A^\top D_a^{-1} A.$$

**Theorem 1 (Adjoint Scale Invariance / Transfer Invariance):**  
For any fine grid $G$ and regional dictionary $\mathcal{R}$ covering all cells ($D_c \ge \mathbf{1}_G$):
$$H \mathbf{1}_G = \mathbf{1}_G.$$

*Proof:*  
For uniform density field $Y = c \mathbf{1}_G$ ($c \in \mathbb{R}$):
1. $A Y = c A \mathbf{1}_G = c D_a \mathbf{1}_M$.
2. $D_a^{-1} (A Y) = c D_a^{-1} D_a \mathbf{1}_M = c \mathbf{1}_M$.
3. $A^\top (c \mathbf{1}_M) = c A^\top \mathbf{1}_M = c D_c \mathbf{1}_G$.
4. $D_c^{-1} (c D_c \mathbf{1}_G) = c \mathbf{1}_G = Y$.  
Hence $H \mathbf{1}_G = \mathbf{1}_G$. $\blacksquare$

**Corollary 1.1 (Scale-Balanced Error Feedback):**  
If the regional head predicts a uniform rate discrepancy $\Delta \rho_m = \delta$, the projected spatial update is identically uniform across all covered cells: $\Delta Y = \delta \mathbf{1}_G$, regardless of the number or size of overlapping regions. It does not imply that uniform density is a trivial fixed point when regional evidence disagrees with spatial predictions ($b \neq A Y$).

---

## 3. Operator-Guided Unrolled Reconciliation (Measure Space)

### 3.1 Regional Energy Geometry
The geometric objective guiding reconciliation is the weighted discrepancy between regional fine sums $q = AY$ and regional head predictions $b$:
$$\mathcal{E}_a(Y) = \frac{1}{2} (AY - b)^\top D_a^{-1} (AY - b) = \frac{1}{2} \sum_{m=1}^M \frac{(q_m - b_m)^2}{|R_m|}.$$

The analytical gradient of $\mathcal{E}_a$ with respect to spatial counts $Y$ is:
$$\nabla_Y \mathcal{E}_a(Y) = A^\top D_a^{-1} (AY - b) = A^\top r^{\text{rate}},$$
where $r_m^{\text{rate}} = \frac{q_m - b_m}{|R_m|}$ is the rate-normalized residual.

### 3.2 Nonnegative Projected SIRT in Measure Space (RMR-P)
Counts must remain strictly non-negative ($Y \ge 0$). In **RMR-P** (the registered benchmark model), updates operate directly in measure space with an exact nonnegativity projection $\Pi_+[x] = \max(x, 0)$:

$$\boxed{Y_{t+1} = \Pi_+ \left[ Y_t - \omega \cdot D_c^{-1} A^\top D_a^{-1} (A Y_t - b) \right], \qquad t = 0, \dots, T-1,}$$

where:
- $T = 2$ unrolled iterations.
- $\omega = 1.0$ is the canonical preconditioned step size.
- Preconditioning by $D_c^{-1}$ scales the adjoint back-projection by cell coverage, ensuring uniform step dynamics across overlapping regions.
- Regional evidence $b$ is detached (`b.detach()`) during solver iterations when evaluating causal reconciliation, decoupling gradient backpropagation from multi-task feature learning.
- Parameter-free: RMR-P introduces **0 additional learnable parameters** over the two-head baseline (B2).

### 3.3 Matched Direct Measure-Space Control (B3b)
To strictly isolate the causal benefit of the exact geometric adjoint $A^\top$ against learned spatial reasoning, the **B3b Learned Projector** is implemented as an exact measure-space counterpart:

$$\boxed{Y_{t+1} = \Pi_+ \left[ Y_t - \omega \cdot P_\theta(F, Y_t, A Y_t - b) \right], \qquad t = 0, \dots, T-1,}$$

where $P_\theta$ is a 3-layer convolutional network conditioning on neck features $F$, current spatial state $Y_t$, and regional count discrepancies $A Y_t - b$. B3b uses identical $T=2, \omega=1.0$, identical solver ramp, identical detached $b$, and operates directly in measure space without any latent Softplus bottleneck.

---

## 4. Architectural Implementation

The architecture matches the concrete implementation in `rmr_count/model.py`:

```
Input Image [1, 3, H, W]
       │
       ▼
MobileNetV4-Conv-Small-0.5 Pretrained Carrier (~87k params)
  ├── Truncated at reduction 16 (C4: 16ch, C8: 32ch, C16: 48ch)
  └── Physically excludes C32 to maximize mobile-edge efficiency
       │
       ▼
Additive FPN Neck (~7.3k params, width=32)
  ├── 1x1 lateral projections of C4, C8, C16 to 32 ch
  ├── Dilated depthwise-separable context (d=1, 2, 3) on P16 prior to additive top-down fusion
  └── Multi-scale feature outputs (P4, P8, P16)
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Measure Head (~3.2k params)               Scale-Matched Regional Head (~4.0k params)
Depthwise-sep Conv3x3 + Conv1x1                Physical scale routing:
Input: P4 (stride 4)                           ├── 32px regions  -> P4
Data-driven prior bias init:                   ├── 64px regions  -> P8
b_0 ≈ -4.1422 (m0 ≈ 0.0158 count/cell)         └── 128px regions -> P16
       │                                       33D Input: 32D visual feature + 1D log(s_R / 32)
       ▼                                              │
Initial Fine Measure Y_0                               ▼
       │                                        Regional Count Evidence b
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            Projected SIRT Reconciliation (T=2, 0 params)
                 │
                 ├── Prefix2D fast integration (FP32): q = A Y_t
                 ├── Rate residual: r = (q - b) / D_a
                 ├── Coverage-normalized adjoint (FP32): r_field = D_c^(-1) A^T r
                 └── Nonnegativity projection: Y_(t+1) = max(Y_t - omega * r_field, 0)
                               │
                               ▼
                     Final Density Field Y_T
```

---

## 5. Registered Experimental Matrix (B0–B5)

Exact parameter counts computed via `RMRCount(variant)` from `rmr_count/model.py`:

| ID | Variant Name | Parameter Count | Regional Head | Unrolled Steps | Operator Adjoint $A^\top$ | Scientific Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 97,681 | ✗ | ✗ | ✗ | Direct regression control without regional reasoning |
| **B1** | Region Loss | 97,681 | ✗ (Loss on $AY$) | ✗ | ✗ | Auxiliary regional rate loss without dual head |
| **B2** | Region Aux | 101,714 | ✓ | ✗ | ✗ | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 100,692 | ✗ | ✓ ($T=2$) | ✗ (Local Conv) | Receptive field expansion via local recurrent refinement |
| **B3b** | Learned Projector | 104,756 | ✓ | ✓ ($T=2$) | ✗ ($P_\theta$ Neural) | Learned neural projection vs exact mathematical adjoint $A^\top$ |
| **B5-P** | **RMR-P (Registered)** | **101,714** | ✓ | ✓ ($T=2$) | ✓ ($H\mathbf{1}=\mathbf{1}$) | Two-step registered RMR-Count model (0 extra params over B2) |

### Core Hypotheses Tested:
1. **$B5\text{-P} > B2$**: Verifies that runtime operator reconciliation provides active spatial correction beyond passive multi-task feature sharing.
2. **$B5\text{-P} > B3b$**: Verifies that the exact mathematical adjoint $A^\top$ outperforms unconstrained learned neural projection in measure space under identical evidence and solver ramp.
3. **$B5\text{-P} > B3a$**: Verifies that regional count conservation provides stronger inductive bias than arbitrary local recurrent refinement.
