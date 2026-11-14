# Scientific Framework & Paper Alignment Matrix (RMRv3)

This document anchors the research formulation, architectural design, baseline positioning, empirical record, and failure-mode disclosure of the **Constrained Radon Measure Recovery (`rmr_v3`)** framework using a four-step survey-driven methodology.

---

## Step 1: Recent Literature Surveys & Architectural Taxonomy (2024–2025)

Rather than deriving research problems from repository artifacts, the framing is grounded in four recent peer-reviewed crowd counting surveys:

| Survey Reference | Scope & Taxonomy | Acknowledged Field Limitations & Open Problems |
| :--- | :--- | :--- |
| **Wang, Zhou, & Chen (2025)**, *IET Image Processing*, `doi:10.1049/ipr2.13328` | Systematically classifies deep crowd counting models into multi-column CNNs, single-column dilated CNNs, attention-augmented CNNs, pure Transformers, and CNN–Transformer hybrids; audits density/point labels, loss functions, and metrics. | Pure ViT encoders suffer quadratic token cost and weak local edge inductive bias; fixed-dilation CNNs struggle with continuous scale variation and non-rigid crowding without dense supervision. |
| **Cheng, Zhai, Li, & Gao (2025)**, *Meas. Sci. Technol.* 36(7), `doi:10.1088/1361-6501/ada69f` | Focuses specifically on **lightweight crowd counting architectures**, public datasets, parameter/latency budgets, and edge deployment constraints. | High-accuracy SOTA models rely on heavy backbones (`> 10M` params) or teacher-student Knowledge Distillation (KD); standalone ultra-lightweight models (`<= 200k` params, `0%` KD) experience severe accuracy degradation in dense crowds and cluttered scenes. |
| **Gao et al. (2025)**, *Vicinagearth*, `doi:10.1007/s44336-024-00011-8` | Comprehensive review of 300+ CNN-based density estimation methods, top-performing pipelines, and standardized density-map/evaluation tooling. | Wide variance in ground-truth map construction and loss engineering obscures which performance gains originate from architecture vs. supervision or dataset-specific tuning. |
| **Supervision Paradigms Survey (2024)** | Covers supervised, semi-supervised, and unsupervised paradigms for crowd counting under point-level supervision. | Point annotations lack spatial extent; translating discrete head points into continuous calibrated mass fields without spatial collapse or background leakage remains mathematically ill-posed. |

---

## Step 2: Dataset-Agnostic Gaps, Problem Statement, RQs, ROs, & Falsifiable Hypotheses

### 2.1 Recognized Field Gaps Addressed
1. **Gap G1 — Efficiency–Spatial Resolution Conflict (`0%` KD, $\le 105k$ Params)**:
   Standard ImageNet-pretrained micro-backbones downsample by $4\times$ to $16\times$. At Stride 4 ($128\times 128$ on a $512\times 512$ input), adjacent heads separated by $< 4\text{px}$ coalesce inside single cells. Recovering fine spatial separation without Transformer self-attention or extra multi-branch parameters requires preserving high-frequency boundary cues inside compact multi-scale feature fusion and depthwise sub-pixel upsampling.
2. **Gap G2 — Ill-Posed Point-to-Measure Equilibrium & Background Fog**:
   Under composite global-count and point-allocation losses, continuous non-negative heads activated by convex Softplus satisfy count targets by distributing predicted mass onto wide, low-amplitude background plateaus where Softplus derivatives decay exponentially ($\sigma(z/\tau) \to 0$ for $z < -6$), causing gradient starvation against weight decay.
3. **Gap G3 — Regional Integral Dilution vs. Direct Neural Measure Estimation**:
   Closing the Radon inverse problem $b = \mathcal{A}_\pi y + \eta$ analytically via closed-form Bregman-Poisson transport (`closed_form_mcbpro_transport`) yields `4.77` MAE on SHA when supplied with Oracle ground-truth box counts (`input_contract.md`), proving mathematical soundness of the operator; however, pooling tiny heads over large boxes dilutes learned regional evidence signals, motivating direct Stride-2 sub-pixel calibration under identical dataset-agnostic priors (`init_m0: auto`).

### 2.2 Formal Problem Statement
Given an image $I \in \mathbb{R}^{3\times H\times W}$ with discrete point annotations $\{p_i\}_{i=1}^N$ and zero external teacher models (`0%` KD), recover a non-negative continuous Radon density field $y \in \mathcal{M}_+(\Omega)$ using $\theta$ with $|\theta| \le 104,441$ trainable parameters and a single unified, dataset-agnostic training configuration, such that total mass $\int_\Omega y(u)\,du \approx N$ and multi-scale linear projections $b = \mathcal{A}_\pi y + \eta$ preserve spatial localization under distribution shift, severe crowding, and image corruptions.

### 2.3 Research Questions (RQs) & Objectives (ROs)

To ensure first-principles scientific framing independent of internal implementation labels, the RQs are formulated directly from Computer Vision, Spatial Statistics, and Inverse Measure Theory open problems, while ROs specify the controlled codebase realizations used to test them:

- **RQ1 (Sub-Pixel Spectral Fusion under Extreme Non-Distilled Capacity)**: Under a strict $\le 105k$ trainable-parameter ceiling with zero teacher distillation, can frequency-decomposed multi-scale feature fusion combined with depthwise sub-pixel upsampling ($H/2 \times W/2$) mitigate dense-head spatial coalescence and quantization error more effectively than coarse Stride-4 dilated context pooling across heterogeneous crowd scales and densities?
  - **RO1**: Implement single-variable neck and resolution ablations comparing sequential depthwise HDC + high-frequency gating (`SFILiteFPNNeck`) + Stride-2 `DWSubpixelMeasureHead` (`100,596` params) against `HDCLiteFPNNeck`, `ASPPLiteNeck`, and canonical Stride-4 under invariant losses, augmentations, and optimizer schedules.
- **RQ2 (Regional Integral Rebalancing vs. Direct Continuous Density Estimation)**: When recovering a continuous Radon measure from noisy multi-scale integral predictions alongside point supervision, does closed-form Kullback-Leibler / Bregman measure projection improve counting and localization over direct feedforward density estimation, or does spatial average-pooling dilution within coarse regions cause learned regional integrals to degrade accuracy relative to the direct fine-grid estimate?
  - **RO2**: Conduct matched Direct vs. learned closed-form transport vs. Oracle ground-truth-integral transport evaluations (`enable_solver: false` vs `enable_solver: true`) under unified supervision to quantify the gap between exact integral physics (`4.77` Oracle MAE) and learned regional evidence.
- **RQ3 (Dataset-Agnostic Prior Calibration & Reliability Under Distribution Shift)**: Can unified prior-density initialization derived strictly from theoretical resolution scaling ($\times 1/s^2$) paired with combined point-likelihood and 1D Sliced Wasserstein (Radon-OT) supervision sustain multi-seed calibration across sparse, dense, corrupted, and zero-shot cross-domain imagery without per-dataset hyperparameter tuning?
  - **RO3**: Benchmark one completely identical `nextgen_sub65` configuration (`init_m0: auto`, fixed double-division bug in [`rmr_v3/engine.py`](file:///f:/lightweightcrcn/rmr_v3/engine.py)) identically across SHA, SHB, UCF-QNRF, and JHU-Crowd++, paired with mandatory reporting of localization (`loc_F1`), corruption sensitivity, scale consistency, distractor false positives, and background-fog mass.

### 2.4 Falsifiable Working Hypothesis
> **H_main**: Under one invariant configuration (`<= 104,441` params, `0%` KD, identical `data` augs / `loss` / `lr` / `ema_decay` across datasets), unifying Stride-2 `DWSubpixelMeasureHead` with `SFILiteFPNNeck` (sequential HDC `[1,2,3]` + high-frequency gating, without ASPP GAP) improves dense crowd resolution and localization over Stride-4 ASPP baselines, while Direct estimation avoids learned regional-evidence dilution relative to `mcbpro` (`MAE_direct <= MAE_mcbpro`) unless regional dilution is solved externally.

Formal statistical thresholds and falsification triggers ($H_1$–$H_5$) are specified in [`input_contract.md`](file:///f:/lightweightcrcn/input_contract.md).

---

## Step 3: Evaluation Ecosystem & Peer-Reviewed Baselines

### 3.1 Datasets & Metrics
- **Benchmarks**: **ShanghaiTech Part A** (`300` train / `182` test), **ShanghaiTech Part B** (`400` train / `316` test), **UCF-QNRF** (`1,201` train / `334` test), **JHU-Crowd++** (`2,272` train / `500` val / `1,600` test w/ Clear, Fog/Haze, Rain, Snow, Distractors), and **UCF_CC_50** (`50` images, 5-fold CV).
- **Metrics**: MAE, RMSE, NAE, Signed Bias, Density-Tier MAE (Sparse / Moderate / Dense), `GAME(0–3)`, Point Localization `loc_F1@4` & `loc_F1@8`, Trainable Parameters, and GFLOPs, summarized on an **Empirical Complexity-Accuracy Trade-off Plot** (*Biểu đồ đánh đổi thực nghiệm độ phức tạp - độ chính xác*).

### 3.2 Peer-Reviewed Literature Reference Table (Verified)

| Model | Venue / Year | KD | Params | GFLOPs | SHA MAE / RMSE | SHB MAE / RMSE | QNRF MAE / RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TinyCount** | JRTIP 2024 (non-KD) | 0% | `0.060M` (`60k`) | `0.16` | `78.2 / 120.8` | — | — |
| **Lw-Count** | TCSVT 2022 | 0% | `0.071M` (`71k`) | — | `69.7 / 100.5` | — | — |
| **RMRv3 (Canonical Stride-4 Anchor, s42 / 2-seed)** | Codebase Canonical | **0%** | **`104,390`** (`~104k`) | — | **`65.62`** (`68.91 ± 4.65` 2-seed) / `111.32` | — | — |
| **RMRv3 (Audited Direct `runs_fixed`)** | Audited Suite | **0%** | **`<= 104,441`** | — | `68.05 / 113.47` (`69.46 ± 1.66` `runs` 3-seed) | `9.24 / 13.92` (Scratch) / `9.63 / 15.37` (PT) | `103.35 / 171.99` |
| **MCNN** | CVPR 2016 | 0% | `0.133M` (`133k`) | `2.80` | `110.2 / —` | — | — |
| **LSANet** | Neurocomputing 2022 | 0% | `0.200M` (`200k`) | `14.4` | `66.1 / 110.2` | `8.6 / —` | — |
| **EBC-ZIP-P** | 2025 | 0% | `0.81M` | `6.46` | `71.18 / 109.60` | `8.23 / 12.62` | `96.29 / 161.82` |
| **EBC-ZIP-N** | 2025 | 0% | `3.36M` | — | `58.86 / 94.63` | — | — |
| **EBC-ZIP-T** | 2025 | 0% | `10.53M` | — | `56.36 / —` | — | — |
| **EBC-ZIP-S** | 2025 | 0% | `33.60M` | — | `55.17 / —` | — | — |
| **EBC-ZIP-B** | 2025 | 0% | `105.60M` | — | `47.81 / —` | — | — |

---

## Step 4: Verified Empirical Results, Unified Protocol & Explicit Failure Modes

### 4.1 Audited Existing Runs (`runs/`, `runs_fixed/`, `model_facts.md`, `input_contract.md`)

All values below are directly measured from on-disk evaluation outputs and audited contract files:

#### A. Canonical Anchor & Solver Probes (`model_facts.md`, `input_contract.md`)
- `rmr_canonical_104k_seed42` (Stride-4 `ASPPLite`, `104,390` params): Raw `65.62 / 111.32`, TTA `65.69 / 109.10`; `seed7`: Raw `72.20`, TTA `72.71`; 2-seed mean: `68.91 ± 4.65`.
- **Oracle Solver Proof (`input_contract.md`)**: `FineHead y0` alone = `65.23 MAE / 108.40 RMSE` vs Full Learned Solver = `65.57 / 111.34` vs Oracle GT-box Solver = `4.77 / 5.78`.
- **Pure Bayesian vs Dual Loss (`input_contract.md`)**: `abl_no_radon_ot` (`lambda_radon_ot: 0.0`, s42) = `65.74 MAE / 108.27 RMSE` (Dense MAE `123.74` vs Canonical `127.67`).

#### B. Audited `runs/benchmarks/` & `runs_fixed/benchmarks/` Suites
| Dataset | Suite | Run ID | MAE | RMSE | Notes / Verified Breakdown |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **SHA** (182 test) | `runs` | Pretrained Direct (s42) | `67.57` | `110.77` | `NAE 0.1652`, `Bias -19.74`, `GAME1 78.25`, `loc_F1@4 0.2954`, `loc_F1@8 0.5820`, 3-seed (`42/1337/2024`): **`69.46 ± 1.66`** |
| **SHA** | `runs` | `ablations_sha/b4_no_neck` (`102,246` p) | `66.24` | `102.27` | Exceeded necked `s42` (`67.57`) in `runs`, whereas failed in `runs_fixed` (`74.78`) |
| **SHA** | `runs_fixed` | `exp_sha_direct_pretrained_104k` | `68.05` | `113.47` | Clean bug-fixed rerun (`exp_sha_pretrained_radon_scaled_104k`: `67.73 / 112.37`; `pafpn_binorm`: `71.65`; `bifpn_lite`: `72.11`; `pconv_fpn`: `76.59`) |
| **SHB** (316 test) | `runs_fixed` | `exp_shb_direct_scratch_104k` | `9.24` | `13.92` | `exp_shb_direct_pretrained_104k`: `9.63 / 15.37`; `bifpn_lite`: `9.69`; `pafpn_binorm`: `9.75` |
| **SHB** | `runs` | `abl_b3_bg_ratio_005` | `9.41` | `14.12` | Pretrained Direct: `10.19 / 15.88` (`Bias -2.33`); Scratch Direct: `10.49 / 16.98` |
| **UCF-QNRF** (334 test) | `runs_fixed` | `exp_qnrf_direct_pretrained_104k` | `103.35` | `171.99` | `Bias -0.11`; `no_neck`: `107.89 / 185.10`; Scratch Direct: `112.96 / 198.34` |
| **UCF-QNRF** | `runs` | `sweeps_train/exp_qnrf_crop_768_104k` | `102.10` | `178.47` | Pretrained Direct: `105.36 / 174.94` (`Bias -5.94`) |
| **JHU-Crowd++** Val (500) | `runs_fixed` | `exp_jhu_direct_pretrained_104k` | `58.07` | `217.04` | `disp10`: `59.73 / 231.14`; `no_neck`: `64.81 / 231.46`; Scratch: `65.20 / 227.23` |
| **JHU-Crowd++** Test (1,600) | `runs` | Pretrained Direct (`attribute_breakdown.json`) | `76.58` | `295.07` | `NAE 0.415`; Clear `71.85` [n=1409], Fog/Haze `100.79` [n=64], Rain `63.35` [n=49], Snow `150.48` [n=78], Distractors `70.74` [n=31] |
| **UCF_CC_50** (5-fold CV) | `runs` | Scratch / Pretrained Direct | `298.65` / `329.99` | — | Std `± 85.18` (Scratch) / `± 100.63` (Pretrained), `ddof=1` |

#### C. Zero-Shot Cross-Dataset Transfers (`runs/eval_*`)
| Source -> Target | MAE | RMSE | Signed Bias |
| :--- | :---: | :---: | :---: |
| `eval_qnrf_on_shb` | `17.71` | `27.12` | `-12.26` |
| `eval_qnrf_on_ucf_cc_50` | `349.28` | `549.59` | `+152.57` |
| `eval_sha_on_shb` | `27.13` | `40.84` | `-17.78` |
| `eval_sha_on_ucf_cc_50` | `411.44` | `635.68` | `-20.33` |

### 4.2 Unified `nextgen_sub65` Protocol (`configs/benchmarks/nextgen_sub65/`) — [Ready to Train]

All 4 primary cross-dataset runs use **100% identical** data augmentations (`crop_size: 512`, `pad_small_images: true`, `max_size: 2048`, `scale_range: [0.6, 1.4]`), **100% identical** model (`sfi_lite`, `hdc_dilations: [1,2,3]`, `use_aspp_gap: false`, `dw_subpixel`, `output_stride: 2`, `subpixel_stride2: true`, `direct`, `init_m0: auto`, `floor_tau: 0.001` -> **`100,596` params**), **100% identical** loss (`dual_bayesian_radon_ot`, `bayesian_sigma: 8.0`, `bayesian_sigma_max: 8.0`, `radon_ot_scale_by_length: false`, `count_nb_dispersion: 50.0`), and **100% identical** optimizer/scheduler (`lr: 0.0003`, `backbone_lr_scale: 0.25`, `warmup_epochs: 15`, `min_lr_ratio: 0.05`, `ema_decay: 0.999`):

| Config File (`configs/benchmarks/nextgen_sub65/`) | Role | Dataset | Epochs | Measured Params | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `exp_sha_direct_sfi_dw_subpixel_102k.yaml` | Proposed Unified Direct | SHA | 1500 | `100,596` | **Ready to Train** |
| `exp_shb_synergistic_hdc_dw_subpixel_100k.yaml` | Proposed Unified Direct | SHB | 1500 | `100,596` | **Ready to Train** |
| `exp_qnrf_synergistic_crop768_dw_subpixel_103k.yaml` | Proposed Unified Direct | QNRF | 1000 | `100,596` | **Ready to Train** |
| `exp_jhu_synergistic_hdc_dw_subpixel_103k.yaml` | Proposed Unified Direct | JHU | 800 | `100,596` | **Ready to Train** |
| `exp_sha_hdc_mcbpro_subpixel_103k.yaml` | Ablation: `hdc_lite` Neck (Direct) | SHA | 1500 | `99,892` | **Ready to Train** |
| `exp_sha_direct_aspp124_dw_subpixel_100k.yaml` | Ablation: `aspp_lite` `[1,2,4]` Neck (Direct) | SHA | 1500 | `99,892` | **Ready to Train** |
| `exp_sha_synergistic_mcbpro_subpixel_103k.yaml` | Ablation: Learned `mcbpro` Solver (`sfi_lite`) | SHA | 1500 | `103,226` | **Ready to Train** |
| `exp_sha_canonical_stride4_mcbpro_repro_104k.yaml` | Ablation: Canonical Stride-4 `ASPPLite+GAP+MCBPRO` | SHA | 1500 | `104,390` | **Ready to Train** |

*(No performance numbers are claimed for these unified Stride-2 `sfi_lite` configs prior to execution; see [`docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md`](file:///f:/lightweightcrcn/docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md) for standalone training commands.)*

### 4.3 Mandatory Failure-Mode & Limitation Registry
1. **Phantom Background Fog ([`results/forensics/full_182_forensic_results.json`](file:///f:/lightweightcrcn/results/forensics/full_182_forensic_results.json))**: `53,528.18` of `81,629.66` total predicted mass (`65.57%`) on SHA test falls outside $5\times 5$ GT head neighborhoods (`sum_gt = 78,862.0`; mean crowd ratio = `30.32%`).
2. **Softplus Bias Saturation**: Background logits freeze near $b \approx -5.442$ because convex Softplus slopes vanish exponentially below $z < -6$, equilibrating against weight decay.
3. **Dense Undercounting & Fine-Structure Attenuation**: Dense SHA scenes (`n=46`) yield `132.16 MAE` on Pretrained Direct (`runs/benchmarks`) and `127.67` on Canonical TTA; snow (`150.48` MAE) and haze (`100.79` MAE) severely degrade JHU test performance.
4. **Distractors & Image Corruptions**: Mean phantom count reaches `185.10` across `4` JHU distractor negatives; JPEG `q=10` raises SHA MAE from `67.91` to `132.63` ([`results/image_corruptions_sweep.json`](file:///f:/lightweightcrcn/results/image_corruptions_sweep.json)); $0.5\times$ rescaling degrades SHA MAE from `67.57` to `116.75` with `10.68%` mean scale consistency error ([`results/scale_invariance_sweep.json`](file:///f:/lightweightcrcn/results/scale_invariance_sweep.json)).
5. **Cross-Dataset Zero-Shot Fragility**: Direct transfer without fine-tuning suffers negative bias under density/scale mismatch (`SHA -> SHB`: `27.13 MAE`, `Bias -17.78`).
