# RMR-v32 Forensic Mistake Registry, Theoretical Grounding & A* Scientific Blueprint

**Document Class:** Permanent Repository Scientific Audit & Theoretical Architecture Guide  
**Status:** Active Canonical Invariant  
**Target Venues:** IEEE TPAMI / CVPR / ICCV  

---

## 1. Executive Mistake Registry & Forensic Accountability (Zero Evasion)

In machine learning research, declaring "no bugs" based on superficial test scripts is the single most destructive anti-pattern. This section records every major historical failure and evasion in the RMR project, the root cause, and the permanent architectural safeguard established to prevent its recurrence.

### 1.1 The Six Historical Evasions & Failure Modes

| # | Bug / Failure Mode | Superficial "No Bugs" Evasion | Forensic Reality & Mathematical Consequence | Permanent Architectural Safeguard |
|---|-------------------|--------------------------------|---------------------------------------------|-----------------------------------|
| **1** | **Dormant Curvature Parameter** (`rmr_core/heads.py`) | "Loss backward passes with no error, curvature loss is active." | Initialized `_CURVATURE_ALPHA_INIT = -8.0`. In PyTorch, $\text{softplus}'(-8.0) = \frac{1}{1 + e^{8.0}} \approx 0.000335$. The effective gradient was attenuated by $>2,980\times$, freezing curvature adaptation for 1,000+ epochs. The module was mathematically dormant. | Calibrated initialization $\alpha_0 \approx -4.0$ matching empirical density prior $m_0 \approx 0.01576$. Mandatory derivative checks ($\partial f / \partial \theta > 10^{-3}$) in unit tests. |
| **2** | **Batch Cross-Talk in Dense Loss Scaling** (`rmr_core/losses/orchestration.py`) | "Loss runs on batch tensors without error." | Normalized dense crop weights across the batch: $w_i = N_i / \sum_j N_j$. This violated **Sample Isolation**: sample 0's gradient depended on sample 1's count. When batch size $B=1$, the scaling collapsed to $1.0 / 1.0 = 1.0$ identically, completely disabling dense loss weighting. | Per-sample scalar computation: $w_i = 1.0 + \min(1.5, \max(0, N_i - \tau) / \tau)$. Sample 0 receives identically zero gradient from Sample 1. |
| **3** | **Uncalibrated Heuristic Threshold** (`configs/rmr_v32/rmr_v32_h4_dense_loss_scaling.yaml`) | "400 is a standard dense threshold in crowd counting." | Threshold was chosen in a vacuum without inspecting the dataset. On ShanghaiTech Part A $512\times 512$ crops, median count is 181, 75th percentile is 357. Exactly 78.5% of crops had count $\le 400$, meaning 4 out of 5 training crops completely bypassed the dense scaling mechanism. | Calibrated against empirical manifest quantiles: $\tau_{\text{dense}} = 250.0$ (covering the top 33% most congested crops). |
| **4** | **Hardcoded Backbone Truncation** (`rmr_core/backbones.py`) | "Multi-backbone support implemented via `MobileNetV4Backbone`." | Hardcoded `self.backbone.blocks[:11]`. When tested on ResNet-50, Swin-Tiny, or ConvNeXt-Femto, the code crashed with `AttributeError: 'ResNet' object has no attribute 'blocks'`. The class claimed multi-backbone support but could only run MobileNet. | Refactored into `TimmPyramidBackbone` using `timm.create_model(..., features_only=True, out_indices=...)`. Tested and verified on MobileNetV4, ResNet-50, Swin-Tiny, and ConvNeXt-Femto. |
| **5** | **Silent PyTorch Broadcasting Trap** (`rmr_core/operators/adjoint.py`, `rmr_v3/solver.py`) | "Unit tests passed on single-element inputs." | In `weighted_normalized_adjoint_field`, `q` was 3D `[B, 1, M]` while `b` was 2D `[B, M]`. In PyTorch, `q - b` does NOT raise an error; it silently broadcasts `(B, 1, M) - (B, M) \to (B, B, M)`. This leaked gradients across batch samples and expanded channel count from 1 to $B$, which silently bypassed basic assertions until Conv2D in TV diffusion crashed with mismatched weights. | Defensive 2D $\to$ 3D unsqueezing at the entry point of all solver and adjoint operators: `if b.ndim == 2: b = b.unsqueeze(1)`. Adversarial batch-gradient isolation tests. |
| **6** | **The Superficial Testing Fallacy** | "39 tests passed, so code is completely bug-free." | Tests used symmetric power-of-two shapes ($512\times 512, 16\times 16$), checked `loss is not None`, and ignored boundary parity, odd resolutions, and mathematical operator duality. | Enacted **Strict Anti-Superficiality Protocol** in `.agents/skills/ai-research-skills/SKILL.md`: mandatory odd resolutions ($409\times 902$), exact adjoint duality $\langle Ay, b \rangle = \langle y, A^\top b \rangle$, and zero-gradient sample isolation. |

---

## 2. Theoretical Re-Evaluation: Proposed Upgrades vs. RMR-v19 (MAE 72.61)

### 2.1 Why Did RMR-v19 Succeed?
RMR-v19 achieved **72.61 TTA MAE / 72.84 Direct MAE** on canonical ShanghaiTech Part A (104,441 parameters, Zero Knowledge Distillation) because every architectural component strictly obeyed physical and mathematical invariants:
1. **Strict Translation Equivariance:** The convolutional backbone (MobileNetV4-Conv-Small truncated) and FPN neck operate purely via local translation-equivariant convolutions. A crowd of 50 people produces identical feature activations regardless of whether it is located at the top, center, or bottom of a crop.
2. **Exact Measure Adjoint Duality:** The unrolled Landweber/SIRT solver used the exact Radon-Nikodym adjoint $A_{\text{RN}}^\top(q - b) = y(x) \sum_{m: x \in R_m} \frac{w_m (q_m - b_m)}{q_m}$. This guaranteed the **Support Absorption Invariant**: background pixels where $y(x) = 0$ receive identically zero residual update.
3. **Barzilai-Borwein BB-1 Spectral Acceleration:** Approximated the inverse Hessian via the Rayleigh quotient on the secant equation:
   $$\alpha_{\text{BB1}} = \frac{\langle s_{k-1}, r_{k-1} \rangle}{\|r_{k-1}\|^2}$$
   providing rapid contraction without manual step size tuning.
4. **Morozov Discrepancy Principle ($\gamma = 0.75$):** Prevented the unrolled solver from overfitting to noisy neural predictions by zeroing out residuals within the noise floor $\gamma \sigma_b$.
5. **Anscombe Variance Stabilization:** Stabilized Poisson counting variance via $\mathcal{L}_{\text{curv}} = (\sqrt{y + \epsilon} - \sqrt{y_{\text{gt}} + \epsilon})^2$, naturally scaling dense gradients by $\sqrt{y_{\text{gt}} / y}$ without heuristic loss scaling.

---

### 2.2 Forensic Theoretical Critique of RMR-v32 Proposed Upgrades

#### Critique 1: Continuous Perspective Carrier Modulation (CPCM - Hypothesis 1)
- **Proposed Mechanism:** $M(u, v) = 1.0 + \tanh(\text{MLP}(u, v))$, with $(u, v) \in [0, 1]^2$ generated via `torch.linspace(0, 1)`.
- **Fatal Theoretical Flaw: Spatial Coordinate Aliasing Under Cropping.**
  During training, the model is trained on random $512\times 512$ crops from images of arbitrary dimensions (e.g. $768\times 1024$).
  - In a crop taken from the top (rows 0 to 512), row 256 has normalized coordinate $v = 0.5$.
  - In a crop taken from the bottom (rows 256 to 768), row 512 has normalized coordinate $v = 0.5$.
  - In physical reality, row 256 is near the camera horizon where human heads have radius $\sim 4$ pixels. Row 512 is in the foreground where heads have radius $\sim 30$ pixels.
  - The MLP `Conv2d(2, 8, 1) -> SiLU -> Conv2d(8, 32, 1)` receives identical coordinates $v = 0.5$ for completely different physical head scales!
  - **Tile Boundary Discontinuity at Test Time:** During tiled inference (`predict_tiled`), each $512\times 512$ tile independently runs CPCM with $v \in [0, 1]$. Across the boundary between Tile 0 (rows 0..512) and Tile 1 (rows 512..1024), row 512 receives modulation $M(v=1.0)$ from Tile 0 and $M(v=0.0)$ from Tile 1. This creates a discontinuous seam across tile boundaries.
- **Scientific Verdict:** CPCM breaks translation equivariance and introduces severe spatial coordinate aliasing. It must NOT be used unless absolute scene-level coordinates $(x / W_{\text{orig}}, y / H_{\text{orig}})$ are explicitly supplied by the data pipeline, and even then, feature-based scale routing (Factorized Routing Head) is theoretically superior because it derives perspective from visual content rather than arbitrary pixel indices.

#### Critique 2: Smooth / Differentiable Floor Suppression vs. MCP Firm Thresholding (Hypothesis 2)
- **Proposed Mechanism:** Replace non-negative proximal thresholding with a smooth function $y_{\text{supp}} = y \cdot \sigma((y - \tau) / \tau)$.
- **Theoretical Flaw: Destruction of Exact Measure Sparsity.**
  The ground truth crowd distribution is a discrete Radon measure $\mu^* = \sum \delta_{x_i}$. In background regions, the true density is identically zero ($y(x) \equiv 0$).
  - In the Radon-Nikodym adjoint update:
    $$[A_{\text{RN}}^\top (q - b)](x) = y(x) \sum_{m: x \in R_m} \frac{w_m (q_m - b_m)}{q_m}$$
    If $y(x) = 0$, the update is identically zero.
  - If a smooth activation function is used, $y(x)$ is strictly positive ($y \approx 10^{-4}$) everywhere.
  - Because $y(x) > 0$, the Radon-Nikodym adjoint NEVER completely suppresses background pixels in subsequent unrolled iterations ($T=6$). Small residual noise continues to be updated and can bloom into phantom mass across large empty regions.
  - In contrast, **Minimax Concave Penalty (MCP) / Firm Thresholding:**
    $$S_{\tau, \mu}(z) = \begin{cases} 0 & z \le \tau \\ \frac{\mu}{\mu - 1}(z - \tau) & \tau < z \le \mu \tau \\ z & z > \mu \tau \end{cases}$$
    achieves exact sparsity ($z \le \tau \implies 0$) while providing an **unbiased identity mapping** ($z > \mu \tau \implies z$) for true crowd peaks, resolving the mass erosion of soft thresholding without leaking background mass.
- **Scientific Verdict:** MCP Firm Thresholding is mathematically superior to smooth floor suppression for Radon measure recovery.

#### Critique 3: Dense Loss Scaling (Hypothesis 4)
- **Proposed Mechanism:** Multiply crop loss by $w = 1.0 + \min(1.5, (N - 250) / 250)$.
- **Theoretical Flaw: Redundant Variance Compounding & Optimizer Disturbance.**
  - Crowd counting follows a Poisson-like counting process where variance scales with the mean: $\text{Var}(N) = \mu$.
  - The Anscombe square-root power loss $\mathcal{L}_{\text{curv}} = (\sqrt{y + c} - \sqrt{y_{\text{gt}} + c})^2$ ALREADY stabilizes Poisson variance! Its gradient:
    $$\frac{\partial \mathcal{L}_{\text{curv}}}{\partial y} = 1 - \sqrt{\frac{y_{\text{gt}} + c}{y + c}}$$
    naturally provides an up to $11\times$ gradient boost on under-counted dense heads.
  - Applying an additional crop-level scalar multiplier $w \in [1.0, 2.5]$ compounds the scaling factor ($w \cdot \sqrt{y_{\text{gt}} / y}$), causing sudden gradient spikes during mini-batch descent that corrupt the running second-moment buffer ($v_t$) of the AdamW optimizer.
  - Most critically, scaling the loss by $2.5\times$ does NOT solve the fundamental physical bottleneck of dense crowds: **the Stride-4 spatial resolution limit.**
- **Scientific Verdict:** Dense loss scaling is an ad-hoc heuristic that distorts optimization moments without addressing the underlying representation resolution.

---

## 3. The True Physical Bottleneck: The Stride-4 Resolution Limit

Why did RMR-v19 achieve 72.61 MAE overall, but dense regions had high error ($\text{MAE}_{\text{dense}} = 127.64$)?
- **Physical Reality:**
  - At Stride 4, a $512\times 512$ image is mapped to a $128\times 128$ grid. Each cell spans $4\times 4 = 16$ image pixels.
  - In congested crowds (e.g. 800 people in a $512\times 512$ crop), adjacent heads are separated by only 2 to 3 pixels.
  - In a Stride-4 grid, two distinct heads separated by 2 pixels fall into the **exact same cell**.
  - A Stride-4 representation physically cannot place two distinct Dirac delta peaks closer than 4 pixels apart.
  - The network is forced to either merge them into a single cell with value 2.0 or smear them across neighboring cells, causing high localized discrepancy.
- **The Genuine Mathematical Solution:**
  - A Sub-pixel Stride-2 Reconstruction Head (via $2\times 2$ PixelShuffle) maps $128\times 128 \times 32$ features to a $256\times 256$ Stride-2 density grid where each cell is $2\times 2$ pixels.
  - This quadruples the spatial Nyquist frequency, allowing heads separated by 2 pixels to be resolved as distinct spatial peaks with less than 40 parameters overhead.

---

## 4. Formal Problem Statement, RQ, RO, and Novelty for A* Publication

### 4.1 Problem Statement
Let $\Omega \subset \mathbb{R}^2$ be a compact spatial domain. Crowd annotations are represented by a discrete Radon measure $\mu^* = \sum_{i=1}^N \delta_{x_i} \in \mathcal{M}_+(\Omega)$. A neural backbone extracts multi-scale continuous feature representations, from which a regional observation operator $A: \mathcal{M}_+(\Omega) \to \mathbb{R}^M$ measures integrated counts over overlapping regions:
$$b_m = [Ay]_m = \int_{R_m} y(x) \, dx$$
Because $M \ll |\Omega|$, inverting $A$ to recover $y(x) \ge 0$ is an ill-posed Fredholm integral equation of the first kind with an infinite-dimensional null space $\mathcal{N}(A)$.

### 4.2 Research Questions (RQ)
- **RQ 1 (Operator Duality & Scale Routing):** How can multi-scale receptive field routing be achieved on the adjoint Hilbert operator ($A_\pi^\top$) using a continuous partition of unity ($\sum \pi_k \equiv 1$) without multi-column parameter bloat or coordinate aliasing?
- **RQ 2 (Measure Adjoint & Support Preservation):** How can residual back-projection be formulated to mathematically guarantee that empty background receives identically zero update ($\text{supp}(y_{t+1}) \subseteq \text{supp}(y_t)$), eliminating background phantom mass accumulation?
- **RQ 3 (Noise Floor Invariance):** How can unrolled Landweber/SIRT iterations regularize against neural prediction noise using the Morozov Discrepancy Principle without over-smoothing sharp crowd peaks?

### 4.3 Research Objectives (RO)
- **RO 1 (Backbone Agnosticism):** The inverse formulation must be modular and pluggable into any modern backbone (MobileNetV4, ResNet-50, Swin-Tiny, ConvNeXt) with zero architectural retraining of the backbone.
- **RO 2 (Benchmark Generalization):** Validate performance across multiple canonical benchmarks (ShanghaiTech Part A & B, UCF-QNRF, NWPU-Crowd) using standard official benchmark protocols (zero ad-hoc splits).
- **RO 3 (Pareto Dominance):** Achieve competitive accuracy with 25M+ parameter models while remaining under $\le 105,000$ trainable parameters and operating at $>60$ FPS on edge hardware.
