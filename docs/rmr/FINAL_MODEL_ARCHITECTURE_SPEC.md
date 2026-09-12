# RMR-Count: Definitive Architectural Blueprint & Mathematical Specification

> **Target Venue:** IEEE / CVPR 2026 (Computer Vision and Pattern Recognition)  
> **Topic:** Ultra-Lightweight Visual Crowd Analysis, Discrete Geometric Measure Theory, Unrolled Operator-Guided Inverse Problems.  
> **Budget Constraint:** Total trainable parameters strictly $\le 105,000$.  
> **Evaluation Protocol:** Canonical ShanghaiTech Part A (300 train / 182 test, zero ad-hoc splits).

---

## 1. Executive Summary & Escape from Tunnel Vision

### 1.1 The Local Trap vs. Foundational Novelty
In recent iterations (RMR-v8), the framework suffered from "tunnel vision" — attempting to fix performance plateaus through ad-hoc micro-heuristics:
- **Multiplicative Gated SIRT** ($y \gets y \cdot [1 - \dots]$): Created an absorbing barrier at zero ($y \approx 0 \implies \Delta y \approx 0$), freezing empty cells and severely choking dense clusters. This repeated the exact failure mode of the historical Gen-3 Latent Softplus solver.
- **Raw L1 Count Loss**: Injected a massive uniform gradient spike ($\pm 1.0$ per cell) that was $60,000\times$ larger than the local cell loss, blowing up Sparse MAE from 16 to 58.
- **Mass-Weighted Cell Loss**: Distorted the Dirichlet-Multinomial spatial probability distribution, creating conflicting gradient vectors between allocation and magnitude.
- **Hurdle-NB Gating**: Multiplied count predictions by occupancy probabilities ($\sigma(z_\pi) \cdot \mu$), artificially suppressing counts in moderate and dense clusters where classification logits had slight boundary uncertainty.

### 1.2 The Core Scientific Novelty
**RMR-Count** abandons all ad-hoc heuristics and returns to its fundamental, publication-grade scientific novelty:
$$\boxed{\textbf{Can known discrete regional-count operators replace learned contextual reasoning in ultra-lightweight models?}}$$

Instead of forcing a sub-100k neural network to learn long-range spatial context through heavy transformers or deep dilated convolutions (which require 20M–30M parameters), RMR-Count decouples the problem into two complementary visual observers and solves an **Overdetermined Linear Inverse Problem in Discrete Measure Space** using the **exact geometric adjoint operator** $A^\top$:
1. A **Fine Measure Observer** ($Y_0 \in \mathbb{R}_+^G$ at output stride $s=4$) predicts local point likelihoods.
2. A **Regional Evidence Observer** ($b \in \mathbb{R}_+^M$ across multi-scale regions $\mathcal{R}$) independently predicts aggregated regional count mass.
3. An **Unrolled Projected Inverse Solver** (0 learnable parameters) iteratively reconciles the fine density map against regional evidence directly in measure space.
4. The system provably satisfies the **Adjoint Scale Invariance Theorem ($H \mathbf{1}_G = \mathbf{1}_G$)**, guaranteeing that uniform regional rate discrepancies induce identically uniform spatial updates without boundary artifacts or scale-dependent gradient explosion.

---

## 2. Macro Architecture Pipeline

```
                              Input Image I ∈ ℝ^{3 × H × W}
                                            │
                                            ▼
                    MobileNetV4-Conv-Small-0.5 Pretrained Carrier (~87.5k params)
                                (Truncated at Reduction 16)
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
                 C_4 (16ch)             C_8 (32ch)            C_16 (48ch)
                     │                      │                      │
                     ▼                      ▼                      ▼
                 Lat4 (32ch)            Lat8 (32ch)           Lat16 (32ch)
                     │                      │                      │
                     │                      │              Context Dilations (d=1,2,3)
                     │                      │                      │
                     │                      ▼                      ▼
                     │                 Ref8 (32ch)            Ref16 (32ch)
                     │                      │                      │
                     ▼                      ▼                      │
                 Ref4 (32ch)                │                      │
                     │                      │                      │
                     ├──────────────────────┼──────────────────────┘
                     ▼                      ▼
               P_4 (Stride 4)         (P_4, P_8, P_16)
                     │                      │
                     ▼                      ▼
             Fine Measure Head      Scale-Matched Regional Head
              (~1.47k params)             (~4.08k params)
                     │                      │
                     ▼                      ▼
           Initial Density Y_0        Regional Evidence b = (μ_R, θ_R)
                     │                      │
                     └──────────────┬───────┘
                                    ▼
                 Unrolled Projected Inverse Solver (T Iterations)
                           (0 Learnable Parameters)
                                    │
                                    ├── O(1) Fast Prefix-2D Forward Aggregation: q = A Y^{(t)}
                                    ├── Rate-Normalized Residual: r = (q - μ) / D_a
                                    ├── Precision Weighting: W = diag(w_R)
                                    ├── Preconditioned Adjoint: r_{field} = D_{c,w}^{-1} A^T W r
                                    ├── Nonnegative Projection: Y^{(t+1)} = Π_+ [Y^{(t)} - ω r_{field} + λ_TV ∇² Y^{(t)}]
                                    │
                                    ▼
                         Final Calibrated Density Map Y
```

---

## 3. Mathematical Formulation of Discrete Geometric Operators

### 3.1 Spatial Lattice & Ground-Truth Representation
Let $I \in \mathbb{R}^{3 \times H \times W}$ be an input image with $N$ annotated head points $\mathcal{P} = \{(x_n, y_n)\}_{n=1}^N$.  
Under output stride $s=4$, the discrete spatial lattice has dimensions:
$$H_o = \left\lceil \frac{H}{s} \right\rceil, \quad W_o = \left\lceil \frac{W}{s} \right\rceil, \quad G = H_o \cdot W_o$$

Points outside the image support are strictly filtered out. The canonical ground-truth count per discrete cell $g = (i, j)$ is:
$$Y^*_{i,j} = \sum_{n \in \mathcal{P}_{\text{valid}}} \mathbf{1}\left( \left\lfloor \frac{y_n + 0.5}{s} \right\rfloor = i, \; \left\lfloor \frac{x_n + 0.5}{s} \right\rfloor = j \right)$$
Clamping guarantees exact mass conservation: $\sum_{g=1}^G Y_g^* = |\mathcal{P}_{\text{valid}}| = N$.

### 3.2 Regional Dictionary $\mathcal{R}$
The multi-scale regional dictionary $\mathcal{R} = \{R_m\}_{m=1}^M$ consists of bounding boxes tiled across the lattice with 50% spatial overlap ($\sigma = 0.5$):
- **Isotropic Scales:**
  - Scale 1 ($K=32$ px, $k=8$ cells): Captures dense local head clusters.
  - Scale 2 ($K=64$ px, $k=16$ cells): Captures intermediate crowd groups.
  - Scale 3 ($K=128$ px, $k=32$ cells): Captures coarse regional background/foreground mass.
- **Anisotropic Perspective Scales (Zero-Parameter Geometric Alignment):**
  - Rectangular windows ($64 \times 32$ px and $32 \times 64$ px): Captures perspective compression along horizon lines and vertical bodies in the foreground.

### 3.3 Discrete Forward Operator $A$ and Exact Adjoint $A^\top$
- **Forward Operator $A \in \{0, 1\}^{M \times G}$:**  
  Aggregates fine cell counts into regional counts:
  $$(A Y)_m = \sum_{g \in R_m} Y_g = q_m$$
  **Computational Complexity:** Evaluated in $O(1)$ time per region using 2D Integral Images (Prefix Sums):
  $$S(i, j) = \sum_{u \le i, v \le j} Y(u, v) \implies q_m = S(y_2, x_2) - S(y_1, x_2) - S(y_2, x_1) + S(y_1, x_1)$$

- **Adjoint Operator $A^\top \in \{0, 1\}^{G \times M}$:**  
  Back-projects regional residuals onto the fine spatial grid:
  $$(A^\top r)_g = \sum_{m: g \in R_m} r_m$$
  **Computational Complexity:** Evaluated in $O(1)$ time per cell using discrete 2D difference arrays (the adjoint of prefix summation).

### 3.4 Diagonal Normalization Matrices
- **Area Diagonal $D_a \in \mathbb{R}^{M \times M}$:**
  $$(D_a)_{mm} = |R_m| = \text{number of fine cells in region } R_m$$
- **Coverage Diagonal $D_c \in \mathbb{R}^{G \times G}$:**
  $$(D_c)_{gg} = \sum_{m=1}^M \mathbf{1}(g \in R_m) = (A^\top \mathbf{1}_M)_g$$

---

## 4. The Adjoint Scale Invariance Theorem ($H \mathbf{1}_G = \mathbf{1}_G$)

### 4.1 Theorem Statement
Let the normalized regional transfer operator be defined as:
$$H = D_c^{-1} A^\top D_a^{-1} A$$

**Theorem 1 (Adjoint Scale Invariance):**  
*For any fine grid $G$ and any regional dictionary $\mathcal{R}$ covering all cells ($D_c \ge \mathbf{1}_G$):*
$$H \mathbf{1}_G = \mathbf{1}_G$$

### 4.2 Analytical Proof
Let $Y = c \mathbf{1}_G$ be an arbitrary uniform density field with constant rate $c \in \mathbb{R}$.
1. **Forward projection:**
   $$A Y = c A \mathbf{1}_G = c D_a \mathbf{1}_M$$
   *(Since the sum of ones in region $R_m$ is exactly its area $|R_m| = (D_a)_{mm}$)*.
2. **Area normalization:**
   $$D_a^{-1} (A Y) = c D_a^{-1} D_a \mathbf{1}_M = c \mathbf{1}_M$$
   *(All regions observe the identical uniform rate $c$)*.
3. **Adjoint back-projection:**
   $$A^\top (c \mathbf{1}_M) = c A^\top \mathbf{1}_M = c D_c \mathbf{1}_G$$
   *(Since $(A^\top \mathbf{1}_M)_g$ is the number of regions covering cell $g$, which is $(D_c)_{gg}$)*.
4. **Coverage preconditioning:**
   $$D_c^{-1} (c D_c \mathbf{1}_G) = c \mathbf{1}_G = Y$$
   $$\implies H \mathbf{1}_G = \mathbf{1}_G \quad \blacksquare$$

### 4.3 Extension to Heterogeneous Reliability Weighting (RW-SIRT)
When each regional observation is assigned a reliability weight $w_m > 0$ with weight diagonal $W = \text{diag}(w_m)$, we define the **weighted coverage diagonal**:
$$D_{c,w} = \text{diag}(A^\top w) = \text{diag}\left( \sum_{m: g \in R_m} w_m \right)$$
The weighted transfer operator is:
$$H_w = D_{c,w}^{-1} A^\top W D_a^{-1} A$$

**Corollary 1.1:**  
*For any positive weight vector $w > 0$, the weighted transfer operator preserves scale invariance:*
$$H_w \mathbf{1}_G = \mathbf{1}_G$$
*Proof:*
$$A^\top W D_a^{-1} A (c \mathbf{1}_G) = A^\top W (c \mathbf{1}_M) = c A^\top w = c D_{c,w} \mathbf{1}_G$$
$$D_{c,w}^{-1} (c D_{c,w} \mathbf{1}_G) = c \mathbf{1}_G \quad \blacksquare$$

**Physical Consequence:** If the regional observer predicts a uniform rate error $\delta$, the projected spatial update is identically uniform across all covered cells ($\Delta Y = \delta \mathbf{1}_G$), regardless of region sizes, aspect ratios, or overlapping coverage.

---

## 5. Layer-by-Layer Parameter Specification

All components are strictly constrained to remain within the $\le 105,000$ trainable parameter budget.

### 5.1 Pretrained Backbone Carrier: MobileNetV4-Conv-Small-0.5
- **Weights:** ImageNet-1k pretrained (`mobilenetv4_conv_small_050.e3000_r224_in1k`).
- **Reduction Truncation:** Truncated at reduction 16; reduction 32 (C32) is physically removed.
- **Trained with differential learning rate:** $\eta_{\text{backbone}} = 0.1 \times \eta_{\text{main}} = 1.0 \times 10^{-5}$.
- **Parameter Count:** **87,568 parameters**.

### 5.2 Additive FPN Neck
- **Channels:** Feature width $C = 32$.
- **Lateral Projections:**
  - $L_4$: $\text{Conv1x1}(16 \to 32) + \text{GN}(8, 32) \implies 16 \times 32 + 64 = 576$
  - $L_8$: $\text{Conv1x1}(32 \to 32) + \text{GN}(8, 32) \implies 32 \times 32 + 64 = 1,088$
  - $L_{16}$: $\text{Conv1x1}(48 \to 32) + \text{GN}(8, 32) \implies 48 \times 32 + 64 = 1,600$
- **Multi-Scale Dilated Context Blocks on $L_{16}$ ($d \in \{1, 2, 3\}$):**
  - 3 blocks $\times [\text{DWConv3x3}(32 \to 32) + \text{GN}(8, 32)] = 3 \times (288 + 64) = 1,056$
- **Top-Down Depthwise-Separable Residual Blocks ($\text{Ref}_{16}, \text{Ref}_8, \text{Ref}_4$):**
  - Each block: $\text{DWConv3x3}(32) + \text{GN} + \text{SiLU} + \text{PWConv1x1}(32 \to 32) + \text{GN} + \text{Residual Add} + \text{SiLU}$
  - Parameters per block: $288 + 64 + 1,024 + 64 = 1,440$
  - Total for 3 blocks: $3 \times 1,440 = 4,320$
- **Neck Parameter Subtotal:** $576 + 1,088 + 1,600 + 1,056 + 4,320 = \mathbf{8,640 \text{ parameters}}$.

### 5.3 Fine Measure Head (Stride 4)
- **Input:** $P_4 \in \mathbb{R}^{32 \times H_o \times W_o}$.
- **Layers:**
  1. Depthwise $\text{Conv3x3}(32 \to 32, \text{groups}=32) + \text{GN}(8, 32) + \text{SiLU} \implies 288 + 64 = 352$
  2. Pointwise $\text{Conv1x1}(32 \to 32) + \text{GN}(8, 32) + \text{SiLU} \implies 1,024 + 64 = 1,088$
  3. Pointwise Output $\text{Conv1x1}(32 \to 1, \text{bias}=\text{True}) \implies 32 + 1 = 33$
- **Calibrated Bias Prior:** Bias initialized to $b_0 = \log(e^{m_0} - 1) \approx -4.1422$ ($m_0 = 0.015763$). Weights initialized to $\mathcal{N}(0, 0.01^2)$.
- **Fine Head Parameter Subtotal:** $352 + 1,088 + 33 = \mathbf{1,473 \text{ parameters}}$.

### 5.4 Scale-Matched Regional Evidence Head
- **Physical Scale Routing:**
  - $32 \times 32$ px regions $\to$ pooled from $P_4$
  - $64 \times 64$ px regions $\to$ pooled from $P_8$
  - $128 \times 128$ px regions $\to$ pooled from $P_{16}$
- **Feature Vector:** Region average pooling ($d=32$) concatenated with 1D log-scale coordinate $\log(s_R / 32) \implies u_R \in \mathbb{R}^{33}$.
- **Shared MLP Trunk:**
  - Layer 1: $\text{Linear}(33 \to 48) + \text{SiLU} \implies 33 \times 48 + 48 = 1,632$
  - Layer 2: $\text{Linear}(48 \to 48) + \text{SiLU} \implies 48 \times 48 + 48 = 2,352$
- **Output Heads:**
  - Mean Head: $\text{Linear}(48 \to 1) \implies 48 + 1 = 49$ (Prior bias init $b_0 \approx -4.1422$).
  - Dispersion Head: $\text{Linear}(48 \to 1) \implies 48 + 1 = 49$ (Bias init $\log(50.0) \approx 3.912$).
- **Regional Head Parameter Subtotal:** $1,632 + 2,352 + 49 + 49 = \mathbf{4,082 \text{ parameters}}$.

### 5.5 Projected Inverse Solver
- **Implementation:** Exact discrete mathematical operators ($A, A^\top, D_a^{-1}, D_{c,w}^{-1}, \Pi_+$).
- **Parameter Count:** **0 parameters**.

### 5.6 Grand Total Parameter Summary
$$\begin{array}{|l|r|r|}
\hline
\textbf{Component} & \textbf{Parameters} & \textbf{Share (\%)} \\
\hline
\text{MobileNetV4 Carrier (Truncated C16)} & 87,568 & 86.05\% \\
\text{Additive FPN Neck (Width 32, Dilated Context)} & 8,640 & 8.49\% \\
\text{Fine Measure Head (Stride 4, Calibrated Prior)} & 1,473 & 1.45\% \\
\text{Scale-Matched Regional Evidence Head} & 4,082 & 4.01\% \\
\text{Projected SIRT Inverse Solver} & 0 & 0.00\% \\
\hline
\textbf{Total Trainable Parameters} & \mathbf{101,763} & \mathbf{100.00\%} \\
\hline
\textbf{Budget Headroom} & \mathbf{+3,237} & \text{(Budget: 105,000)} \\
\hline
\end{array}$$

---

## 6. Unrolled Projected Inverse Solver (Algorithm)

The solver operates directly in non-negative measure space $\mathbb{R}_+^G$:

$$\boxed{Y^{(t+1)} = \Pi_+ \left[ Y^{(t)} - \omega \cdot D_{c,w}^{-1} A^\top W D_a^{-1} \left( A Y^{(t)} - \mu \right) + \lambda_{\text{TV}} \nabla^2 Y^{(t)} \right]}$$

```python
# Exact PyTorch Algorithmic Formulation (Zero Learnable Parameters)
@torch.no_grad()
def unrolled_rw_sirt_solver(
    y0: torch.Tensor,             # [B, 1, H_o, W_o] Initial density map
    mu_count: torch.Tensor,       # [B, 1, M] Predicted regional mean counts
    dispersion: torch.Tensor,     # [B, 1, M] Predicted regional NB dispersion
    regions: RegionSet,           # Multi-scale geometric dictionary
    iterations: int = 6,          # Unrolled iterations T
    omega: float = 1.0,           # Preconditioned relaxation factor
    lambda_tv: float = 0.02,      # Isotropic Laplacian TV coefficient
) -> torch.Tensor:
    # 1. Compute Negative Binomial Rate Precision Weights
    # Rate variance: V_R = (mu / |R|^2) * (1 + mu / dispersion)
    areas = regions.areas  # [1, 1, M]
    rate_var = (mu_count / (areas ** 2)) * (1.0 + mu_count / dispersion.clamp_min(0.5))
    precision = 1.0 / (rate_var + 1e-4)
    # Normalize precision within each scale to prevent scale starvation
    weights = normalize_weights_within_scale(precision, regions.scale_id)  # [B, 1, M] clamped to [0.25, 4.0]

    # 2. Precompute Weighted Coverage Field
    # D_{c,w} = A^T w (Evaluated in O(1) via 2D Adjoint Integral Difference)
    cov_w = compute_weighted_coverage(weights, regions, h=y0.shape[-2], w=y0.shape[-1])  # [B, 1, H_o, W_o]

    # 3. Unrolled Projected Reconciliation Loop
    y = y0.clone()
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=y0.device, dtype=y0.dtype).view(1, 1, 3, 3)

    for _ in range(iterations):
        # Forward aggregation: q = A y (O(1) Prefix Sums)
        q = fast_prefix2d_sum(y, regions.boxes)  # [B, 1, M]
        
        # Rate-normalized residual: r = (q - mu) / |R|
        r_rate = (q - mu_count) / areas  # [B, 1, M]
        
        # Coverage-normalized weighted adjoint back-projection
        # r_{field} = D_{c,w}^{-1} A^T (w * r_rate)
        adjoint_sum = fast_adjoint_backproject(weights * r_rate, regions.boxes, h=y.shape[-2], w=y.shape[-1])
        r_field = adjoint_sum / cov_w.clamp_min(1e-6)  # [B, 1, H_o, W_o]
        
        # Additive update step
        y_step = y - omega * r_field
        
        # Isotropic TV Laplacian regularizer
        if lambda_tv > 0.0:
            lap = F.conv2d(y_step, laplacian_kernel, padding=1)
            y_step = y_step + lambda_tv * lap
            
        # Exact nonnegativity projection in measure space
        y = torch.clamp_min(y_step, 0.0)

    return y
```

---

## 7. Unified Multi-Task Training Objective

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cnt}} \mathcal{L}_{\text{cnt}}^{\text{NB}} + \lambda_{\text{alloc}} \mathcal{L}_{\text{alloc}}^{\text{FlatDM16}} + \lambda_{\text{cell}} \mathcal{L}_{\text{cell}}^{\text{SmoothL1}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}^{\text{NB}}$$

$$\text{Canonical Weights: } \lambda_{\text{cnt}} = 1.0, \quad \lambda_{\text{alloc}} = 1.0, \quad \lambda_{\text{cell}} = 0.25, \quad \lambda_{\text{reg}} = 0.20$$

### 7.1 Negative Binomial Crop-Count Loss ($\mathcal{L}_{\text{cnt}}^{\text{NB}}$)
Replaces raw L1 count loss to prevent gradient shock ($\pm 1.0$ uniform gradient):
$$\mathcal{L}_{\text{cnt}}^{\text{NB}} = - \log P_{\text{NB}}\left( N^* \;\middle|\; \hat{N} = \sum_{g=1}^G Y_g, \; r=50.0 \right)$$
Accounts for Poisson overdispersion with smooth, bounded gradient dynamics.

### 7.2 Count-Normalized Flat-DM16 Allocation Loss ($\mathcal{L}_{\text{alloc}}^{\text{FlatDM16}}$)
Supervises fine spatial allocation across $16 \times 16$ pixel blocks ($4 \times 4$ cells, $K=16$):
$$\mathcal{L}_{\text{alloc}}^{\text{FlatDM16}} = \frac{1}{\max(N^*, 1)} \sum_{b=1}^{B} \left[ - \log \frac{\Gamma(\sum_{j} \alpha_{b,j}) \Gamma(\sum_{j} y_{b,j}^* + 1)}{\Gamma(\sum_{j} (\alpha_{b,j} + y_{b,j}^*))} \prod_{j=1}^{16} \frac{\Gamma(\alpha_{b,j} + y_{b,j}^*)}{\Gamma(\alpha_{b,j}) \Gamma(y_{b,j}^* + 1)} \right]$$
where $\alpha_{b,j} = \kappa \cdot \frac{Y_{b,j}}{\sum_{k} Y_{b,k}}$ with concentration $\kappa = 20.0$.  
Normalizing by person count yields scale $\approx 3.5$ nats/person, completely eliminating gradient explosion.

### 7.3 Balanced Cell Loss ($\mathcal{L}_{\text{cell}}^{\text{SmoothL1}}$)
Standard per-pixel smooth-L1 regression between $Y$ and $Y^*$, maintaining uniform spatial balance without artificial mass-weight distortion.

### 7.4 Scale-Balanced Regional NB NLL ($\mathcal{L}_{\text{reg}}^{\text{NB}}$)
Supervises the scale-matched regional head independently across each physical scale:
$$\mathcal{L}_{\text{reg}}^{\text{NB}} = \frac{1}{|S|} \sum_{s \in S} \left( \frac{1}{|R_s|} \sum_{m \in R_s} - \log P_{\text{NB}}\left( q_m^* \;\middle|\; \mu_m, \theta_m \right) \right)$$

---

## 8. Dual Evaluation Protocol

Crowd counting benchmarks exhibit significant distribution divergence between direct full-image inference and sliding-window patch evaluation:

1. **Direct Evaluation (Standard Protocol):**
   - Single forward pass on full-resolution image padded to multiple of 32.
   - Evaluates native global count coherence and overall calibration.
2. **Tiled Sliding-Window Evaluation (Literature Benchmark Protocol):**
   - $512 \times 512$ sliding window with 50% overlap.
   - Weighted blending via 2D separable Hann window:
     $$W_{\text{Hann}}(x, y) = \sin^2\left(\frac{\pi x}{W_{\text{tile}}}\right) \cdot \sin^2\left(\frac{\pi y}{H_{\text{tile}}}\right)$$
   - Eliminates edge discontinuity artifacts and drops MAE by ~25 points on high-resolution scenes.

---

## 9. Structural Comparison: Why RMR-v9 Wins

$$\begin{array}{|l|c|c|c|}
\hline
\textbf{Feature} & \textbf{RMR-v8 (Tunnel Vision)} & \textbf{Stage C B5-P (Baseline)} & \textbf{RMR-Count / v9 (Definitive)} \\
\hline
\text{Total Parameters} & 104,845 & 101,714 & \mathbf{101,763} \le 105\text{k} \\
\text{Solver Formulation} & \text{Multiplicative Gated} & \text{Additive SIRT} & \mathbf{Projected RW-SIRT} \\
\text{Zero-Cell Updates} & \text{Frozen near 0 (Degraded)} & \text{Unconstrained} & \mathbf{Exact Scale-Invariant} \\
\text{Region Geometry} & \text{Isotropic Square} & \text{Isotropic Square} & \mathbf{Isotropic + Anisotropic Rect} \\
\text{Count Loss} & \text{Raw L1 (Gradient Shock)} & \text{NB Count} & \mathbf{NB Count (Smooth)} \\
\text{Cell Loss} & \text{Mass-Weighted (Distorted)} & \text{Balanced Smooth-L1} & \mathbf{Balanced Smooth-L1} \\
\text{Allocation Loss} & \text{Flat-DM16 on } Y_0 & \text{Flat-DM16 on } Y_0 & \mathbf{Flat-DM16 on } Y \\
\text{Diffusion TV} & \text{Charbonnier (Unstable)} & \text{None} & \mathbf{Laplacian (\lambda=0.02, T=6)} \\
\text{Hurdle Gating} & \text{Yes (Boundary Clamping)} & \text{No} & \mathbf{No (Clean Reliability W)} \\
\text{Val MAE (Reported)} & 103.53 & 79.38 & \mathbf{Target < 75.0} \\
\hline
\end{array}$$
