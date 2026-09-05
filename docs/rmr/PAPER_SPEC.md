# RMR-Count: Regional Measure Reconciliation for Ultra-Lightweight Crowd Counting

> **Target Venue:** IEEE / CVPR 2026 (Computer Vision and Pattern Recognition)  
> **Topic:** Ultra-Lightweight Visual Crowd Analysis, Operator-Guided Optimization Layers, Discrete Geometric Measure Theory.  
> **Scope:** Pure visual crowd counting and spatial density estimation (< 100k parameters). This is an explicit counting architecture; point localization, detection bounding boxes, and Hungarian matching are out of scope.

---

## 1. Abstract & Motivation

In ultra-lightweight crowd counting (< 100k parameters), models must maintain local spatial resolution while preserving global count coherence under strict memory and computational constraints. Heavy crowd counters rely on deep transformer backbones, dense self-attention, or large multi-scale dilated convolutions to integrate long-range context. However, these mechanisms exceed the parameter and compute budget of mobile-edge carriers.

We investigate a fundamental research question:
$$\boxed{\textbf{Can known discrete regional-count operators replace part of learned contextual reasoning in ultra-lightweight models?}}$$

**RMR-Count** (Regional Measure Reconciliation) answers this question through a decoupled, operator-guided architecture:
1. A **Fine Density Head** estimates initial spatial cell densities $y_0 \in \mathbb{R}_+^G$ at output stride $s=4$.
2. A **Regional Extensivity Head** independently predicts aggregated count mass $b \in \mathbb{R}_+^M$ across multi-scale bounding regions $\mathcal{R}$.
3. An **Operator-Guided Unrolled Reconciliation Layer** iteratively reconciles local densities against regional count constraints using the **exact geometric adjoint operator** $A^\top$.

Crucially, we prove that under the area-normalized and coverage-preconditioned transfer operator $H = D_c^{-1} A^\top D_a^{-1} A$, the system satisfies the **Adjoint Scale Invariance Theorem**: $H \mathbf{1}_G = \mathbf{1}_G$. This guarantees that uniform density fields remain unaltered across arbitrary multi-scale partitions, preventing scale-dependent gradient explosion or boundary artifacts.

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

**Theorem 1 (Adjoint Scale Invariance):**  
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
If the regional head predicts a uniform rate discrepancy $\Delta \rho_m = \delta$, the projected spatial update is identically uniform across all covered cells: $\Delta Y = \delta \mathbf{1}_G$, regardless of the number or size of overlapping regions.

---

## 3. Operator-Guided Unrolled Reconciliation

### 3.1 Regional Energy Geometry
The geometric objective guiding reconciliation is the weighted discrepancy between regional fine sums $q = AY$ and regional head predictions $b$:
$$\mathcal{E}_a(Y) = \frac{1}{2} (AY - b)^\top D_a^{-1} (AY - b) = \frac{1}{2} \sum_{m=1}^M \frac{(q_m - b_m)^2}{|R_m|}.$$

The analytical gradient of $\mathcal{E}_a$ with respect to spatial counts $Y$ is:
$$\nabla_Y \mathcal{E}_a(Y) = A^\top D_a^{-1} (AY - b) = A^\top r^{\text{rate}},$$
where $r_m^{\text{rate}} = \frac{q_m - b_m}{|R_m|}$ is the rate-normalized residual.

### 3.2 Unrolled Latent Updates (RMR-Latent vs RMR-Jacobian)
Counts must remain strictly non-negative ($Y \ge 0$). We parameterize cell densities in latent space $z \in \mathbb{R}^G$ via $Y = \operatorname{softplus}(z)$.

In **RMR-Latent** (the registered benchmark model), the spatial field:
$$r^{\text{field}} = \operatorname{clip}\left(D_c^{-1} A^\top D_a^{-1} (AY - b), \; [-\tau, \tau]\right)$$
is derived from regional-energy geometry and applied as an explicit reconciliation direction directly in latent log-density space $z$:

$$\boxed{z^{(t+1)} = z^{(t)} - \eta_t \cdot M^{(t)} \cdot r^{\text{field}}, \qquad Y^{(t+1)} = \operatorname{softplus}(z^{(t+1)}),}$$

where:
- $\eta_t = \eta_{\text{max}} \cdot \sigma(\alpha_t)$ is a parameterized step size with learnable logits $\alpha_t$, initialized to $\eta_t(0) = \eta_{\text{init}} = 0.05$ (with $\eta_{\text{max}} = 0.20$).
- $M^{(t)}$ is a lightweight feature preconditioner gating updates based on local semantic context.
- $\tau = 5.0$ clips extreme rate residuals to ensure numerical stability.

*Remark on Optimization Geometry:* RMR-Latent deliberately omits the sigmoid Jacobian factor $\sigma(z) \approx 0.01$ associated with chain-rule gradient descent $\nabla_z \mathcal{E}_a = \sigma(z) \nabla_Y \mathcal{E}_a$. Including $\sigma(z)$ suppresses updates ~100x on sparse cells. RMR-Jacobian is retained strictly as an ablation (`use_jacobian_gate: true`).

---

## 4. Architectural Implementation

The architecture matches the concrete implementation in `rmr_count/model.py`:

```
Input Image [1, 3, H, W]
       │
       ▼
TinyLocalEncoder (~52k params)
  ├── Stem: ConvGNAct(3 -> 16, k=3, stride=2)
  ├── s4:   TinyIR(16 -> 24, stride=2) + TinyIR(24 -> 24)
  ├── s8:   TinyIR(24 -> 40, stride=2) + 2x TinyIR(40 -> 40)
  └── s16:  TinyIR(40 -> 64, stride=2) + TinyIR(64 -> 64)
       │
       ▼
AdditiveFusion Neck (~3.5k params, width=32)
  ├── 1x1 Projections of C4, C8, C16 to 32 ch
  ├── Bilinear upsampling to C4 resolution + elementwise addition
  └── Fused via depthwise-separable 3x3 ConvGNAct + 1x1 ConvGNAct
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Density Head (~3.2k params)               Regional Extensivity Head (~4.2k params)
Depthwise-sep Conv3x3 + Conv1x1                ROI-Pooling on {32, 64, 128} px
init bias: b_0 ≈ -4.595 (softplus ≈ 0.01)      + 4D Geometry [log h, log w, log A, log(w/h)]
       │                                        init: b_R = |R| * rho_R (calibrated to 0.01)
       ▼                                               │
Initial Latent Field z_0                               │
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
            Unrolled Regional Reconciliation (T=2, ~1.5k params)
                 │
                 ├── Prefix2D fast integration: q = A Y^(t)
                 ├── Rate residual: r = (q - b) / D_a
                 ├── Adjoint back-projection: r_field = D_c^(-1) A^T r
                 ├── Latent update: z^(t+1) = z^(t) - eta_t * M^(t) * r_field
                 └── Re-activation: Y^(t+1) = softplus(z^(t+1))
                               │
                               ▼
                     Final Density Field Y_T
```

---

## 5. Registered Experimental Matrix (B0–B5)

Exact parameter counts computed via `count_parameters(model)` from `rmr_count/model.py`:

| ID | Variant Name | Parameter Count | Regional Head | Unrolled Steps | Operator Adjoint $A^\top$ | Scientific Hypothesis Tested |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B0** | Direct Baseline | 58,867 | ✗ | ✗ | ✗ | Direct regression control without regional reasoning |
| **B1** | Region Loss | 58,867 | ✗ (Loss on $AY$) | ✗ | ✗ | Auxiliary regional rate loss without dual head |
| **B2** | Region Aux | 63,044 | ✓ | ✗ | ✗ | Multi-task dual head without runtime reconciliation |
| **B3a** | Local Refine | 61,876 | ✗ | ✓ ($T=2$) | ✗ (Unconstrained) | Receptive field expansion via local recurrent refinement |
| **B3b** | Learned Projector | 66,086 | ✓ | ✓ ($T=2$) | ✗ (Blackbox MLP) | Learned neural projection vs exact mathematical adjoint $A^\top$ |
| **B4** | RMR-T1 | 64,580 | ✓ | ✓ ($T=1$) | ✓ ($H\mathbf{1}=\mathbf{1}$) | Single-step unrolled reconciliation |
| **B5** | **RMR-T2 (Registered)** | **64,581** | ✓ | ✓ ($T=2$) | ✓ ($H\mathbf{1}=\mathbf{1}$) | Two-step registered RMR-Count model |

### Core Hypotheses Tested:
1. **$B5 > B2$**: Verifies that runtime operator reconciliation provides active spatial correction beyond passive multi-task feature sharing.
2. **$B5 > B3b$**: Verifies that the exact mathematical adjoint $A^\top$ outperforms unconstrained learned neural projection.
3. **$B5 > B3a$**: Verifies that regional count conservation provides stronger inductive bias than arbitrary recurrent refinement.
