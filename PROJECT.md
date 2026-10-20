# Project: RMR (Radon Measure Recovery) Ultra-Lightweight Crowd Counting

## 1. Project Overview & Empirical Invariants
The RMR framework formulates crowd counting on ShanghaiTech Part A (canonical split: 300 train / 182 test images) as an inverse problem in measure spaces, solving:
$$b = \mathcal{A}_\pi y + \eta$$
$$y_{t+1} = \mathcal{P}_+\left(y_t + \omega_t \frac{y_t}{\text{cov}_\pi} \mathcal{A}_\pi^T \mathcal{S}_{\gamma \sigma}(b - \mathcal{A}_\pi y_t)\right)$$

### Non-Negotiable Invariants:
1. **Parameter Budgets**:
   - **Track A (Stride 4 Anchor)**: Strictly $\le 104,441$ trainable parameters.
   - **Track B (Subpixel Stride 2 Capacity Extension)**: Strictly $\le 200,000$ trainable parameters.
2. **100% Standalone**: Exactly $0.0\%$ Knowledge Distillation (zero teacher models at training or inference).
3. **Empirical Integrity**: Zero fabricated metrics, zero fake oracles. Evaluated on the full 182 test images using official MAE / RMSE / GAME(0,1,2,3) / NAE protocols.
4. **Code Quality**: Strictly $\le 450$ lines per Python file. Pure Python / YAML configurations.
5. **Optimization Invariant**: Standard Cosine Annealing (1,000 epochs, warmup 10, min_lr_ratio 0.05). WSD (Warmup-Stable-Decay) is permanently banned and removed.

---

## 2. Unified System Architecture

### 2.1. Backbone Network
- `MobileNetV4Backbone` ([`rmr_core/backbones.py`](file:///f:/lightweightcrcn/rmr_core/backbones.py)): Truncated `mobilenetv4_conv_small_050` (87,568 parameters) producing multi-scale representations at strides 4, 8, and 16.

### 2.2. Multi-Scale Neck Topology
- **Track A (ASPP-Lite Neck)**: Width 32, dilations $\{1, 3, 6\}$ + Global Average Pooling (10,784 params) unifying context into carrier $P_4$.
- **Track B (RepASPP-Lite Neck)**: Structural re-parameterization with depthwise $7 \times 7$ convolutions, fusing multi-scale feature hierarchies within the $\le 200,000$ parameter budget.

### 2.3. Output Heads
- **Fine Density Head** ([`rmr_core/heads.py`](file:///f:/lightweightcrcn/rmr_core/heads.py)): Generates continuous density measure $y_0$ (Stride 4 or Subpixel Stride 2 PixelShuffle) with Softplus activation and analytical negative background bias ($b_0 \approx -4.1422$).
- **Probabilistic Regional Evidence Head** ([`rmr_v3/regional_head.py`](file:///f:/lightweightcrcn/rmr_v3/regional_head.py)): Predicts multi-scale regional counts $\mu_r$ and dispersion $\kappa_r$ across spatial boxes $[32, 64, 128]\text{px}$.
- **Scale Routing Head** ([`rmr_core/scale_routing.py`](file:///f:/lightweightcrcn/rmr_core/scale_routing.py)): Predicts continuous scale assignment probabilities $\pi_k(u) \in \Delta^2$.

### 2.4. Unrolled Inverse Solver (SIRT)
- $T=6$ differentiable iterations ([`rmr_v3/solver.py`](file:///f:/lightweightcrcn/rmr_v3/solver.py)).
- Pure Barzilai-Borwein BB-1 Rayleigh contraction step sizes.
- Radon-Nikodym adjoint transfer guaranteeing support absorption ($\text{supp}(y_{t+1}) \subseteq \text{supp}(y_0)$).
- Spatially adaptive Morozov discrepancy deadbands preventing noise amplification.

### 2.5. Purified Synergistic Loss Formulation
- Orthogonal 3-part loss structure avoiding gradient conflicts:
  1. Microscopic point allocation: Canonical Bayesian Loss or Flat-DM16 optimal transport.
  2. Macroscopic regional count: Beta-weighted Negative-Binomial NLL.
  3. Density-balanced cell supervision: Fractional-power normalized cell loss ($p=0.5$).

---

## 3. Directory Structure & Key Entry Points

```text
lightweightcrcn/
├── configs/
│   ├── rmr_v50/            # Active Track A / Track B experimental configurations
│   ├── rmr_research/       # Core research & historical benchmark suites
│   └── rmr_sub60/          # Canonical Sub-60 anchor configurations
├── data/
│   ├── part_A_final/       # ShanghaiTech Part A benchmark images & annotations
│   ├── sha_a_train_all.jsonl # 300 official training samples
│   └── sha_a_test.jsonl    # 182 official test samples
├── docs/                   # Authoritative technical documentation and audits
│   └── code_explanation/  # Modular walkthroughs of architecture, solver, loss, and training
├── rmr_core/               # Shared canonical primitives (backbones, necks, heads, operators)
├── rmr_v3/                 # Active research engine (trainer, solver, loss orchestration)
│   ├── config/             # Strict schema validation & provenance tracking
│   ├── losses/             # Purified loss modules (allocation, point supervision, cell)
│   ├── model/              # Unified RMR architecture definitions
│   └── optim/              # Optimizer & scheduler factories (Cosine Annealing, Schedule-Free)
├── tests/                  # Automated verification test suite
└── train.py                # Root training entry point (python train.py --config <config_path>)
```
