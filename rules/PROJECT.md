# Project: RMR (Radon Measure Recovery) Ultra-Lightweight Crowd Counting

## 1. Project Overview & Non-Negotiable Invariants

The RMR framework formulates crowd counting on ShanghaiTech Part A (canonical split: 300 train / 182 test images) as an ill-posed inverse problem in continuous Radon measure spaces:
$$b = \mathcal{A}_\pi y + \eta$$
$$y^* = y_0 \cdot \mathcal{A}_\pi^T \left( \frac{b}{\mathcal{A}_\pi y_0 + \epsilon} \right)$$

### Non-Negotiable Research Invariants:
1. **Strict Parameter Ceilings**:
   - **Track A (Stride 4 Anchor)**: Strictly $\le 104,441$ trainable parameters.
   - **Track B (Subpixel Stride 2 Capacity Extension)**: Strictly $\le 200,000$ trainable parameters.
2. **100% Standalone (Zero Knowledge Distillation)**:
   - Exactly $0.0\%$ Knowledge Distillation (zero teacher models at training or inference).
3. **Empirical Integrity**:
   - Zero fabricated metrics, zero oracle substitutions in reported benchmarks. Full evaluation on all 182 test images using official MAE / RMSE / GAME(0,1,2,3) / NAE protocols.
4. **Code Quality**:
   - Strictly $\le 450$ lines per Python file. Pure Python / YAML configurations.
5. **Zero Frankencode & Anti-Patchwork Invariant**:
   - Strictly forbidden to arbitrarily cobble together heuristics or ad-hoc patches. Everything must have rigorous mathematical/physical justification.
6. **Controlled Experimentation Protocol**:
   - Zero random experiments. All proposals require explicit mathematical justification against champion baselines (V64 MAE 69.03, V72 MAE 70.02).

---

## 2. Champion Benchmarks & Baseline Status

All models trained on official ShanghaiTech Part A (300 train images) and evaluated on all 182 test images with exactly 0.0% Knowledge Distillation:

| Run Identifier | Track | Stride | Parameters | Neck Topology | Allocation Loss | Solver Mode | Val MAE | Val RMSE | Significance & Benchmark Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`v69_exp1_track_a_dual_bayesian_radon_ot_1500ep`** | **Track A** | 4 | **104,390** | RepASPPLite | Dual Bay + Radon-OT | mcbpro | **65.33** | **108.39** | **ALL-TIME BEST TRACK A CHAMPION (TTA)** ($\le 104,441$ budget) |
| **`v70_track_b_exp4b_no_solver`** | **Track B** | 2 | **122,618** | RepASPPLite | Flat DM16 | direct (no solver) | **65.69** | **108.08** | **ALL-TIME BEST TRACK B CHAMPION** ($\le 200,000$ budget) |
| `v69_exp1_track_a_dual_bayesian_radon_ot_1500ep` | Track A | 4 | 104,390 | RepASPPLite | Dual Bay + Radon-OT | mcbpro | **65.82** | **110.47** | Track A standard validation checkpoint |
| `v70_track_b_exp4b_no_solver` | Track B | 2 | 122,618 | RepASPPLite | Flat DM16 | direct (no solver) | **65.70** | **108.04** | Track B tiled inference benchmark |
| `v64_champion_synergy_50mae_1500ep` | Track B | 2 | 197,839 | RepASPPLite | Flat DM16 | mcbpro | **69.03** | **118.24** | Verified V64 Anchor Baseline |
| `v72_track_b_exp0_v64_anchor_stride2` | Track B | 2 | 197,839 | RepASPPLite | Flat DM16 | mcbpro | **70.02** | **107.86** | Active Track B V72 Reproduction Checkpoint |
| `v67_exp4_abl_pure_2loss_no_dm16_1500ep` | Track B | 2 | 197,839 | RepASPPLite | Pure 2-Loss (Bay + Count) | mcbpro | **70.85** | **111.97** | Elimination of Flat DM16 conflict |
| `v65_champion_pure_synergy_1500ep` | Track B | 2 | 197,839 | RepASPPLite | Dual Bay + DM16 | mcbpro | 73.12 | 129.54 | Dual Allocation Ablation |
| `v63_champion_aspp_subpixel_1500ep` | Track B | 2 | 125,786 | ASPPLite | Flat DM16 | mcbpro | 74.54 | 134.12 | Initial Stride 2 Subpixel Milestone |
| `v50_track_a_repaspp` | Track A | 4 | 104,359 | RepASPPLite | Flat DM16 | mcbpro | 82.41 | 145.20 | Track A Parameter Baseline |

---

## 3. Unified System Architecture

### 3.1. Encoder Backbone
- `TimmPyramidBackbone` ([`rmr_core/backbones.py`](file:///f:/lightweightcrcn/rmr_core/backbones.py)): Truncated `mobilenetv4_conv_small_050` (87,568 parameters) producing multi-scale representations at strides 4 ($C_4$), 8 ($C_8$), and 16 ($C_{16}$).

### 3.2. Multi-Scale Neck Topologies
- **Track A (ASPPLite Neck)**: Width 32, dilations $\{1, 3, 6\}$ + Global Average Pooling unifying context into carrier $P_4$.
- **Track B (RepASPPLite Neck)**: Structural re-parameterization with depthwise $7 \times 7$ convolutions, fusing multi-scale feature hierarchies within the $\le 200,000$ parameter budget.

### 3.3. Fine Carrier Density Head
- `FineMeasureHead` ([`rmr_core/heads.py`](file:///f:/lightweightcrcn/rmr_core/heads.py)): Generates continuous density measure $y_0$ (Stride 4 or Subpixel Stride 2 PixelShuffle) with Softplus activation and analytical negative background bias ($b_0 \approx -4.1422$ for Stride 4; $b_0 \approx -5.442$ for Stride 2).

### 3.4. Probabilistic Regional Evidence Head
- `ProbabilisticRegionalEvidenceHead` ([`rmr_v3/regional_head.py`](file:///f:/lightweightcrcn/rmr_v3/regional_head.py)): Predicts multi-scale regional counts $\mu_r$ and dispersion $r_r$ across spatial boxes $[32, 64, 128]\text{px}$.

### 3.5. Inverse Measure Solver
- `closed_form_mcbpro_transport` ([`rmr_core/operators/mcbpro.py`](file:///f:/lightweightcrcn/rmr_core/operators/mcbpro.py)): Non-iterative closed-form Bregman-Poisson I-projection enforcing regional mass conservation.

---

## 4. Primary Root Cause of the Sub-50 MAE Bottleneck

Exhaustive dataset-wide audit across all 182 test images ([`full_182_forensic_results.json`](file:///f:/lightweightcrcn/scratch/full_182_forensic_results.json)) has mathematically established:
1. **Shortcut Learning Equilibrium**:
   Under composite count and allocation losses, the model counts by splitting mass into $\approx 35.6\%$ on true heads and $\approx 64.4\%$ into a uniform background fog ($\approx 0.00124\text{ count/cell}$).
2. **Convex Activation Trap**:
   Because Softplus derivative decays exponentially ($\sigma(z) \to 0$ for $z < -6$), gradient descent cannot push background logits below $-7.7$ against weight decay, freezing the background mass at $\approx 218$ phantom people on full images.
3. **Regional Head Area Dilution**:
   Average-pooling over large boxes dilutes individual micro-heads by $99.5\%$, destroying the linear correlation between predicted box counts and ground truth ($r = 0.0521$).
4. **Resolution Path**:
   Transforming continuous density support into compact measure support with decoupled presence modeling to force 100% of the mass onto head peaks.
