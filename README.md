# Constrained Radon Measure Recovery (`rmr_v3`)

Ultra-lightweight, $0.0\%$ Knowledge Distillation crowd counting under a strict parameter ceiling ($\le 104,441$ trainable parameters), formulated as Constrained Radon Measure Recovery across diverse density regimes (**ShanghaiTech Part A/B**, **UCF-QNRF**, **JHU-Crowd++**, and **UCF_CC_50**).

---

## 1. Mathematical Formulation & Core Scope

Point-supervised crowd counting seeks a non-negative continuous measure $\mu \in \mathcal{M}_+(\Omega)$ with density $y(u) \ge 0$ on $\Omega \subset \mathbb{R}^2$ governed by multi-scale linear integral projections:
$$b = \mathcal{A}_\pi y + \eta, \qquad N_{\text{pred}} = \int_\Omega y(u)\,du.$$

- **Parameter Budget**: Strictly $\le 104,441$ trainable parameters ($100,596$ params in the unified Direct configuration).
- **Standalone (`0.0%` KD)**: Zero teacher networks or pre-computed teacher density distillation during training or inference.
- **Unified Dataset-Agnostic Design (`configs/benchmarks/nextgen_sub65/`)**: Identical model topology, `init_m0: auto` density calibration, data augmentations, composite loss (`Dual Bayesian + 4-angle Radon-OT + Negative-Binomial Count`), and optimizer/EMA hyperparameters across SHA, SHB, UCF-QNRF, and JHU-Crowd++.

---

## 2. Unified Sub-$105k$ Architecture (`100,596` Trainable Params)

| Stage | Module (`rmr_core/` & `rmr_v3/`) | Role | Trainable Params |
| :--- | :--- | :--- | :---: |
| **Backbone** | `TimmPyramidBackbone` ([`rmr_core/backbones.py`](file:///f:/lightweightcrcn/rmr_core/backbones.py)) | Truncated `mobilenetv4_conv_small_050.e3000_r224_in1k` producing $(C_4, C_8, C_{16})$ | `87,568` |
| **Neck** | `SFILiteFPNNeck` ([`rmr_core/necks/sfi.py`](file:///f:/lightweightcrcn/rmr_core/necks/sfi.py)) | Lateral FPN fusion + sequential HDC (`[1,2,3]`) + $3\times 3$ high-frequency decomposition gating (`use_aspp_gap: false`) | `10,432` |
| **Fine Carrier Head** | `DWSubpixelMeasureHead` ([`rmr_core/heads.py`](file:///f:/lightweightcrcn/rmr_core/heads.py)) | Depthwise $3\times 3$ separable head + `PixelShuffle(2)` upsampling to Stride-2 $y_0$, Softplus activation, `init_m0: auto` | `2,596` |
| **Total (Direct Mode)** | `enable_regional_head: false`, `enable_solver: false` | Single-pass continuous measure output at $H/2 \times W/2$ | **`100,596`** |

Optional closed-form regional transport (`ProbabilisticRegionalEvidenceHead` + [`closed_form_mcbpro_transport`](file:///f:/lightweightcrcn/rmr_core/operators/mcbpro.py), `103,226` total params with `sfi_lite` + `dw_subpixel`) is retained specifically for single-variable solver ablations against Direct mode.

---

## 3. Repository Structure

```text
f:\lightweightcrcn
├── AGENTS.md                        # Repository-wide agent & terminology rules
├── README.md                        # Overview & quickstart
├── input_contract.md                # Falsifiable hypotheses (H1–H5) & negative result archive
├── model_facts.md                   # Exact module parameter breakdown & mechanism audits
├── rmr_core/                        # Backbones, necks (sfi/hdc/aspp/rep), heads, closed-form operators
├── rmr_v3/                          # Engine, config schema/validator, losses, trainer & eval CLI
├── configs/benchmarks/nextgen_sub65/ # Unified cross-dataset & single-variable ablation YAMLs
├── docs/
│   ├── SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md
│   └── CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md
├── rules/
│   ├── PROJECT.md                   # Architecture & invariant summary
│   ├── GEMINI.md                    # Environment & research guidelines
│   ├── RULES.md                     # Hard engineering & scientific rules
│   ├── DERIVATION_PACKAGE.md        # Closed-form transport & Beta-NLL derivations
│   └── CHANGELOG.md                 # Audited version & fix log
└── tests/                           # Unit & contract tests (core, rmr_v3, rmr_v23)
```

---

## 4. Audited Benchmark Summary & Explicit Failure Modes

### Verified Existing Benchmark Runs (On-Disk Logs)
The metrics below come directly from completed evaluation outputs (`runs/`, `runs_fixed/`, `model_facts.md`, and `input_contract.md`). The unified Stride-2 `sfi_lite` configs in [`configs/benchmarks/nextgen_sub65/`](file:///f:/lightweightcrcn/configs/benchmarks/nextgen_sub65/) are validated and **Ready to Train** (no numbers are claimed prior to GPU execution).

| Benchmark | Audited Config / Run | Output Stride | MAE | RMSE | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **SHA (182 test)** | Canonical `rmr_canonical_104k_seed42` (`104,390` p) | 4 | `65.62` | `111.32` | `seed7: 72.20` (2-seed mean `68.91 ± 4.65`); Oracle GT-box solver: `4.77` |
| **SHA (182 test)** | `runs/benchmarks` Pretrained Direct (3-seed) | 4 | `69.46 ± 1.66` | — | `s42: 67.57 / 110.77`, `s1337: 70.19`, `s2024: 70.64`; `b4_no_neck`: `66.24` |
| **SHA (182 test)** | `runs_fixed` `exp_sha_direct_pretrained_104k` | 4 | `68.05` | `113.47` | `radon_scaled`: `67.73 / 112.37`; `no_neck`: `74.78` |
| **SHB (316 test)** | `runs_fixed` `exp_shb_direct_scratch_104k` / PT | 4 | `9.24` / `9.63` | `13.92` / `15.37` | `runs` `abl_b3_bg_ratio_005`: `9.41 / 14.12` |
| **UCF-QNRF (334 test)** | `runs_fixed` `exp_qnrf_direct_pretrained_104k` | 4 | `103.35` | `171.99` | `Bias -0.11`; `runs` `exp_qnrf_crop_768_104k`: `102.10 / 178.47` |
| **JHU-Crowd++ Val (500)** | `runs_fixed` `exp_jhu_direct_pretrained_104k` | 4 | `58.07` | `217.04` | Official Test (`1,600` imgs, `runs`): `76.58 / 295.07 / NAE 0.415` |
| **UCF_CC_50 (5-fold CV)** | `runs/benchmarks` Scratch / Pretrained Direct | 4 | `298.65` / `329.99` | — | 5-fold std `± 85.18` (Scratch) / `± 100.63` (Pretrained), `ddof=1` |

### Documented Technical Failure Modes
1. **Phantom Background Fog ([`results/forensics/full_182_forensic_results.json`](file:///f:/lightweightcrcn/results/forensics/full_182_forensic_results.json))**: `53,528.18` of `81,629.66` predicted mass (`65.57%`) on SHA sits outside $5\times 5$ GT head neighborhoods.
2. **Softplus Bias Freeze**: Background logits stagnate near $b \approx -5.442$ because convex Softplus gradients vanish exponentially below $z < -6$ against weight decay.
3. **Dense Undercounting & Regional Pooling Dilution**: Dense SHA images (`n=46`) suffer `132.16` MAE on Pretrained Direct (`runs/benchmarks`); learned `mcbpro` slightly degrades raw `y0` (`65.57` vs `65.23`) due to average pooling inside large boxes.
4. **Distractor & Corruption Sensitivity**: $4$ JHU distractor negatives produce `185.10` mean phantom count; JPEG `q=10` increases SHA MAE to `132.63`; $0.5\times$ rescaling raises MAE to `116.75`.
5. **Cross-Dataset Zero-Shot Shift**: Zero-shot `SHA -> SHB` drops to `27.13` MAE (`Bias -17.78`, `runs/eval_sha_on_shb`).

---

## 5. Standalone CLI Quickstart

Run tests (Windows PowerShell):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/core/test_cross_dataset_universality.py tests/core/test_non_aspp_necks.py tests/rmr_v3/test_radon_ot.py tests/rmr_v23/test_rmr_v23.py
```

Train the unified Stride-2 `SFILite + DWSubpixel` configuration on SHA (`1500` epochs):
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml
```

Evaluate a trained checkpoint (standard + localization metrics):
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.eval --config configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml --checkpoint runs_fixed/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k/best_val_mae.pt --eval-localization
```

For all 8 unified cross-dataset and ablation commands (SHA, SHB, UCF-QNRF, JHU-Crowd++), see [`docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md`](file:///f:/lightweightcrcn/docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md) and [`docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md`](file:///f:/lightweightcrcn/docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md).
