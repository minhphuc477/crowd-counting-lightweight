# Project: Constrained Radon Measure Recovery (RMRv3) for Ultra-Lightweight Crowd Counting

## 1. Project Overview & Mathematical Formulation

The RMRv3 repository investigates **Constrained Radon Measure Recovery** for standalone ($0.0\%$ Knowledge Distillation) crowd counting under a strict parameter ceiling ($\le 104,441$ trainable parameters) across diverse crowd density regimes (**ShanghaiTech Part A**, **ShanghaiTech Part B**, **UCF-QNRF**, and **JHU-Crowd++**).

Mathematically, point-supervised crowd counting seeks a non-negative continuous Radon measure $\mu \in \mathcal{M}_+(\Omega)$ with Lebesgue density $y(u) \ge 0$ on $\Omega \subset \mathbb{R}^2$ such that linear integral observations over spatial windows $\{R_k\}$ satisfy:
$$b = \mathcal{A}_\pi y + \eta, \qquad N_{\text{pred}} = \int_\Omega y(u) \, du.$$

### Core Design & Operational Invariants
1. **Strict Parameter Ceiling & 0% KD**: Strictly $\le 104,441$ trainable parameters with $0.0\%$ Knowledge Distillation (no teacher networks during training or inference).
2. **Dataset-Agnostic Configuration**: Zero dataset-specific architecture hacks, hardcoded logit floors, or bespoke per-dataset hyperparameters in the unified `configs/benchmarks/nextgen_sub65/` suite. Resolution priors auto-calibrate via `init_m0: auto` and analytical Softplus bias initialization ($\tau_{\text{floor}} = \text{Softplus}^{-1}(m_{0,\text{eff}})$).
3. **Empirical Integrity & Anti-Hype (`AGENTS.md`)**: Every reported metric must come directly from audited `summary.json`, `attribute_breakdown.json`, `results/*.json`, or peer-reviewed literature tables. Strictly follow `AGENTS.md` Rule 1 (banned terminology); always use *"Empirical Complexity-Accuracy Trade-off"*.
4. **Code & Script Invariants**: Keep Python modules $\le 450$ lines. Never generate or edit automated batch runner scripts (`train_all.ps1`, `run_all.sh`, etc.); always present standalone CLI commands in code blocks.

---

## 2. Active System Architecture (`rmr_core/` & `rmr_v3/`)

### Unified Sub-$105k$ Configuration (`nextgen_sub65`)
- **Encoder Backbone ([`rmr_core/backbones.py`](file:///f:/lightweightcrcn/rmr_core/backbones.py))**: Truncated `mobilenetv4_conv_small_050.e3000_r224_in1k` (`87,568` params) outputting pyramid features $(C_4, C_8, C_{16})$.
- **Multi-Scale Neck ([`rmr_core/necks/sfi.py`](file:///f:/lightweightcrcn/rmr_core/necks/sfi.py))**: `SFILiteFPNNeck` (`neck_type: sfi_lite`), combining top-down FPN lateral fusion (`lat4/8/16`), sequential depthwise Hybrid Dilated Convolutions (`hdc_dilations: [1, 2, 3]`, zero gridding holes), spatial high-frequency decomposition ($F_{\text{HF}} = F - \text{AvgPool}_{3\times 3}(F)$), and multi-scale selective gating (`ref4/8/16`), without ASPP Global Average Pooling (`use_aspp_gap: false`). Also supports clean single-variable comparisons against `hdc_lite`, `aspp_lite`, `bifpn_lite`, `pafpn_binorm`, and `pconv_fpn` ([`rmr_v3/model/builder.py`](file:///f:/lightweightcrcn/rmr_v3/model/builder.py)).
- **Subpixel Carrier Head ([`rmr_core/heads.py`](file:///f:/lightweightcrcn/rmr_core/heads.py))**: `DWSubpixelMeasureHead` (`head_type: dw_subpixel`, `output_stride: 2`, `subpixel_stride2: true`, `head_activation: softplus`) applying depthwise $3\times 3$ separable refinement followed by zero-parameter `PixelShuffle(2)` upsampling to output $y_0$ at $H/2 \times W/2$ resolution, with internal $m_0 / 4.0$ calibration synced across both [`rmr_v3/trainer.py`](file:///f:/lightweightcrcn/rmr_v3/trainer.py) and [`rmr_v3/engine.py`](file:///f:/lightweightcrcn/rmr_v3/engine.py).
- **Direct Mode vs. Regional Solver (`mcbpro`)**: The unified baseline operates in Direct mode (`enable_regional_head: false`, `enable_solver: false`, `solver_mode: direct`). Optional `ProbabilisticRegionalEvidenceHead` + closed-form [`closed_form_mcbpro_transport`](file:///f:/lightweightcrcn/rmr_core/operators/mcbpro.py) remain toggleable for controlled ablation against Direct mode.
- **Composite Supervision (`rmr_v3/losses/`)**: `dual_bayesian_radon_ot` combining `BayesianPointLoss` ($\lambda = 0.15$), 4-angle `RadonOTLoss` ($\lambda = 0.85$), and Negative-Binomial global count loss ($\lambda = 0.5$, $r = 50.0$).

---

## 3. Verified Benchmarks & Known Failure Modes

### Verified Existing Benchmark Runs (On-Disk Logs)
- **Canonical SHA Anchor (`rmr_canonical_104k_seed42`, Stride-4 MCBPRO, 104,390 params)**: Raw `65.62 MAE / 111.32 RMSE` (`seed7`: `72.20`, 2-seed mean `68.91 +/- 4.65`); TTA `65.69 / 109.10`; Oracle-GT-box MCBPRO drops error to `4.77 / 5.78` whereas learned `FineHead y0` alone achieves `65.23 / 108.40` vs Full solver `65.57 / 111.34`.
- **Audited Multi-Dataset Suite (`runs_fixed/benchmarks/`, Stride-4)**:
  - SHA Pretrained Direct (`exp_sha_direct_pretrained_104k`): `68.05 MAE / 113.47 RMSE` (`runs/benchmarks` 3-seed mean: `69.46 +/- 1.66`).
  - SHB Scratch Direct (`exp_shb_direct_scratch_104k`): `9.24 MAE / 13.92 RMSE` (Pretrained Direct: `9.63 / 15.37`).
  - UCF-QNRF Pretrained Direct (`exp_qnrf_direct_pretrained_104k`): `103.35 MAE / 171.99 RMSE` (`Bias -0.11`; `runs/benchmarks/sweeps_train/exp_qnrf_crop_768_104k`: `102.10 / 178.47`).
  - JHU-Crowd++ Val Pretrained Direct (`exp_jhu_direct_pretrained_104k`, 500 imgs): `58.07 MAE / 217.04 RMSE` (`runs/benchmarks` Official Test, 1,600 imgs: `76.58 / 295.07 / NAE 0.415`).
  - UCF_CC_50 (5-fold CV, `runs/benchmarks`): Scratch `298.65 +/- 85.18`, Pretrained `329.99 +/- 100.63`.

### Explicit Technical Failure Modes (Mandatory Reporting)
1. **Phantom Background Fog ([`results/forensics/full_182_forensic_results.json`](file:///f:/lightweightcrcn/results/forensics/full_182_forensic_results.json))**: Across SHA's 182 test images, `sum_y0 = 81,629.66` vs `sum_gt = 78,862.0`, with `53,528.18` mass (`65.57%`) residing outside $5\times 5$ GT head neighborhoods.
2. **Softplus Bias Freeze**: Under convex Softplus activation ($y = \tau \log(1 + e^{z/\tau})$), background logit gradients vanish exponentially for $z < -6$, equilibrating against weight decay near $b \approx -5.442$.
3. **Dense Undercounting & Regional Dilution**: Dense subset MAE reaches `132.16` on SHA Pretrained (`runs/benchmarks`, $n=46$ dense images) and `127.67` on Canonical TTA; spatial pooling in large boxes dilutes micro-heads, causing learned regional evidence to underperform $y_0$ alone.
4. **Domain Shift, Distractors & Corruptions**: Zero-shot cross-dataset evaluation degrades severely (`runs/eval_sha_on_shb`: `27.13`, `runs/eval_qnrf_on_shb`: `17.71`); $4$ JHU distractor scenes yield `185.10` mean phantom count; JPEG `q=10` increases SHA MAE from `67.91` to `132.63`; $0.5\times$ rescaling degrades SHA MAE from `67.57` to `116.75`.

See [`docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md`](file:///f:/lightweightcrcn/docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md) and [`docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md`](file:///f:/lightweightcrcn/docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md) for the complete literature framework and standalone training commands.
