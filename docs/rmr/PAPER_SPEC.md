# RMR-Count: Regional Measure Regularization for Ultra-Lightweight Crowd Counting

> **Target Venue:** IEEE / CVPR 2026 (Computer Vision and Pattern Recognition)  
> **Topic:** Ultra-Lightweight Visual Crowd Analysis, Implicit Optimization Layers, Discrete Geometric Measure Theory.

---

## 1. Abstract & Executive Summary

Ultra-lightweight crowd counting (< 100k parameters) faces a fundamental architectural dilemma: mobile-edge carriers (e.g., MobileNetV4-Conv-Small-0.50) possess small effective receptive fields that fail to capture long-range contextual dependencies, leading to massive overcounting in clutters and undercounting in sparse backgrounds. While global transformer necks or dense multi-scale dialations mitigate context deficits in heavy networks, they exceed the parameter and compute budget of mobile hardware.

**RMR-Count** resolves this dilemma through a decoupled dual-head framework governed by an exact discrete geometric operator algebra:
1. A **Fine Density Head** predicts local initial cell densities $y_0 \in \mathbb{R}_+^G$ at output stride $s=4$.
2. A **Regional Extensivity Head** directly estimates aggregated count mass $b \in \mathbb{R}_+^M$ over multi-scale bounding regions $\mathcal{R}$.
3. An **Implicit Unrolled Solver** iteratively reconciles local fine densities against regional constraints via the exact adjoint operator $A^\top$, minimizing the regional energy $\mathcal{E}_a(Y) = \frac{1}{2}(AY - b)^\top D_a^{-1} (AY - b)$ without backpropagating through large spatial graphs or dense attention maps.

Crucially, we prove that under the area-normalized and coverage-preconditioned operator $H = D_c^{-1} A^\top D_a^{-1} A$, the system satisfies the **Adjoint Scale Invariance Theorem**: $H \mathbf{1}_G = \mathbf{1}_G$. This guarantees that uniform density distributions remain unaltered across arbitrary regional partitions, preventing scale-dependent gradient explosion or boundary artifacts.

---

## 2. Mathematical Formulation

### 2.1 Spatial Discretization & Cell Representation
Let an input image $I \in \mathbb{R}^{3 \times H \times W}$ contain $N$ annotated head points $\mathcal{P} = \{(x_n, y_n)\}_{n=1}^N$.
Under output stride $s=4$, the spatial counting lattice has dimensions $H_o = \lceil H/s \rceil, W_o = \lceil W/s \rceil$ with total grid cells $G = H_o \cdot W_o$.

The discrete ground-truth count per cell is:
$$Y_{ij}^* = \sum_{n=1}^N \mathbf{1}\left(\left\lfloor \frac{y_n}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n}{s} \right\rfloor = j\right), \qquad \sum_{i,j} Y_{ij}^* = N.$$

### 2.2 Regional Projection Operator $A$ and Adjoint $A^\top$
We define a multi-scale regional dictionary $\mathcal{R} = \{R_m\}_{m=1}^M$ spanning window sizes $K \in \{32, 64, 128\}$ pixels (grid dimensions $k = K/s \in \{8, 16, 32\}$) with stride overlap $\sigma = 0.5$.

The forward regional projection matrix $A \in \{0, 1\}^{M \times G}$ aggregates cell counts into regional masses:
$$(A Y)_m = \sum_{g \in R_m} Y_g = q_m.$$

The adjoint operator $A^\top \in \{0, 1\}^{G \times M}$ scatters regional residual feedback back onto the fine spatial grid:
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

**Corollary 1.1 (Uniform Error Preservation):**  
If the regional head predicts a uniform rate discrepancy $\Delta \rho_m = \delta$, the projected spatial update is identically uniform across the entire scene: $\Delta Y = \delta \mathbf{1}_G$.

---

## 3. RMR Unrolled Solver Dynamics

### 3.1 Regional Count Energy
The objective of the solver is to align local predictions $Y$ with regional evidence $b$:
$$\mathcal{E}_a(Y) = \frac{1}{2} (AY - b)^\top D_a^{-1} (AY - b) = \frac{1}{2} \sum_{m=1}^M \frac{(q_m - b_m)^2}{|R_m|}.$$

The analytical gradient of $\mathcal{E}_a$ with respect to spatial cell counts $Y$ is:
$$\nabla_Y \mathcal{E}_a(Y) = A^\top D_a^{-1} (AY - b) = A^\top r^{\text{rate}},$$
where $r_m^{\text{rate}} = \frac{q_m - b_m}{|R_m|}$ is the rate-normalized residual.

### 3.2 Preconditioned Latent Updates
Direct updates on non-negative counts $Y$ risk violating physical positivity ($Y \ge 0$). We parameterize $Y = \operatorname{softplus}(z)$ and perform unrolled gradient steps in latent log-density space $z \in \mathbb{R}^G$:

$$z^{(t+1)} = z^{(t)} - \eta_t \cdot M^{(t)} \cdot \operatorname{clip}\left(D_c^{-1} A^\top D_a^{-1} (A Y^{(t)} - b), \; [-\tau, \tau]\right),$$
where:
- $\eta_t = \eta_{\text{init}} + (\eta_{\text{max}} - \eta_{\text{init}}) \cdot \frac{t}{T}$ is the step-size schedule ($\eta_{\text{init}}=0.05, \eta_{\text{max}}=0.20$).
- $M^{(t)}$ is a dynamic feature preconditioner gating updates based on local semantic confidence.
- $\tau = 5.0$ clips extreme rate residuals, preventing divergence on dense outliers.
- $Y^{(t+1)} = \operatorname{softplus}(z^{(t+1)})$.

In the registered model (**RMR-Latent**), $M^{(t)}$ operates directly on $z$ without sigmoid Jacobian damping, avoiding gradient vanishing when cells have small count values.

---

## 4. Architectural Implementation

```
Input Image [1, 3, H, W]
       │
       ▼
MobileNetV4-Conv-Small-0.50 (truncated C16, ~65k params)
       │
       ▼
Additive FPN Neck (32 channels, context dilations {1, 2, 3})
       │
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Fine Density Head                               Regional Extensivity Head
Conv3x3 -> GN -> SiLU -> Conv1x1                ROI-Pooling on {32, 64, 128} px
init: b_0 ≈ softplus(-4.595) ≈ 0.01             + 4D Geometry [log h, log w, log A, log(w/h)]
       │                                        init: b_R = |R| * rho_R (calibrated to 0.01)
       ▼                                               │
Initial Latent Field z_0                               │
       │                                               │
       └───────────────────────┬───────────────────────┘
                               ▼
               RMR Implicit Unrolled Solver (T=2)
                 │
                 ├── Prefix2D fast regional integration: q = A Y^(t)
                 ├── Compute rate residual: r = (q - b) / D_a
                 ├── Adjoint back-projection: r_field = D_c^(-1) A^T r
                 ├── Latent update: z^(t+1) = z^(t) - eta_t * M^(t) * r_field
                 └── Re-activation: Y^(t+1) = softplus(z^(t+1))
                               │
                               ▼
                     Final Density Field Y_T
```

---

## 5. Registered Experimental Matrix (B0–B5)

To decisively demonstrate the necessity of each component, we benchmark 6 scientific variants across identical training schedules (60 epochs, warmup 5, AdamW 3e-4, 3 random seeds):

| ID | Variant Name | Parameter Count | Regional Head | Iterative Solver | Conservation Theorem |
|:---|:---|:---:|:---:|:---:|:---:|
| **B0** | Direct Baseline | 64,581 | ✗ | ✗ | ✗ |
| **B1** | Region Loss (Single Head) | 64,581 | ✗ (Aux loss on $AY$) | ✗ | ✗ |
| **B2** | Region Aux (Dual Head) | 82,143 | ✓ (Loss supervision) | ✗ | ✗ |
| **B3a** | Local Refine (Iterative GRU) | 88,412 | ✗ | ✓ (Unconstrained) | ✗ |
| **B3b** | Learned Projector | 94,820 | ✓ | ✓ (Blackbox MLP) | ✗ |
| **B4** | RMR-T1 (1 Iteration) | 82,143 | ✓ | ✓ ($T=1$) | ✓ ($H\mathbf{1}=\mathbf{1}$) |
| **B5** | **RMR-T2 (Registered Model)** | **82,143** | ✓ | ✓ ($T=2$) | ✓ ($H\mathbf{1}=\mathbf{1}$) |

### Core Hypotheses Tested:
1. **$B5 > B2$**: Verifies that runtime iterative projection is superior to passive multi-task feature sharing.
2. **$B5 > B3b$**: Verifies that the exact mathematical adjoint $A^\top$ outperforms learned, unconstrained neural projection networks.
3. **$B5 > B3a$**: Verifies that regional measure conservation provides stronger spatial regularization than arbitrary recurrent refinement.
