# Codebase Entrypoints & Operations Manual (`rmr_v3`)

This manual documents the active modules, schema-validated configuration contracts, unit tests, and standalone CLI commands for **Constrained Radon Measure Recovery (`rmr_v3`)**.

---

## 1. Active Architecture & Core Modules

| Component | File Path | Symbols & Verified Function |
| :--- | :--- | :--- |
| **Backbone** | [`rmr_core/backbones.py`](file:///f:/lightweightcrcn/rmr_core/backbones.py) | `TimmPyramidBackbone` (`mobilenetv4_conv_small_050.e3000_r224_in1k`, `87,568` params) extracting strides `4/8/16`. |
| **Neck Modules** | [`rmr_core/necks/sfi.py`](file:///f:/lightweightcrcn/rmr_core/necks/sfi.py) | `SFILiteFPNNeck` (`sfi_lite`, `10,432` params): lateral projection + sequential HDC (`[1,2,3]`) + `avg_pool2d(3,1,1)` high-frequency gating. |
| **Neck Registry** | [`rmr_v3/model/builder.py`](file:///f:/lightweightcrcn/rmr_v3/model/builder.py) | `build_neck_module` supporting `sfi_lite`, `hdc_lite`, `aspp_lite`, `additive`, `rlk_hdc`, `rep_weighted`, `rep_aspp_lite`, `bifpn_lite`, `pafpn_binorm`, `pconv_fpn`. |
| **Measure Heads** | [`rmr_core/heads.py`](file:///f:/lightweightcrcn/rmr_core/heads.py) | `FineMeasureHead` (Stride-4 `1,473` p) & `DWSubpixelMeasureHead` (`dw_subpixel`, `2,596` p, `PixelShuffle(2)`). Both divide `m0` by `4.0` internally when `subpixel_stride2=True`. |
| **Closed-Form Transport** | [`rmr_core/operators/mcbpro.py`](file:///f:/lightweightcrcn/rmr_core/operators/mcbpro.py) | `closed_form_mcbpro_transport` (0-parameter closed-form Bregman-Poisson I-projection when `enable_solver: true`). |
| **Model Construction** | [`rmr_v3/engine.py`](file:///f:/lightweightcrcn/rmr_v3/engine.py) | `make_model(cfg)` resolves `"auto"` `init_m0` via unmodified `compute_manifest_density(..., output_stride=4)` matching [`rmr_v3/trainer.py`](file:///f:/lightweightcrcn/rmr_v3/trainer.py). |
| **Config Validation** | [`rmr_v3/config/schema.py`](file:///f:/lightweightcrcn/rmr_v3/config/schema.py), [`rmr_v3/config/validator.py`](file:///f:/lightweightcrcn/rmr_v3/config/validator.py), [`rmr_v3/config/cross_checks.py`](file:///f:/lightweightcrcn/rmr_v3/config/cross_checks.py) | `validate_v3_config(cfg)` enforces allowed keys (`data`, `model`, `loss`, `train`, `eval`) and cross-section constraints (`enable_regional_head: false` requires `enable_solver: false` and `lambda_region_nb <= 0.0`). |

---

## 2. Verified CLI Entrypoints & Flags

### Training (`rmr_v3.trainer`)
```text
python -m rmr_v3.trainer -c CONFIG [-r RESUME] [--seed SEED] [--lr LR]
                         [--output-dir OUTPUT_DIR] [--save-dir SAVE_DIR] [--run-id RUN_ID]
                         [--device DEVICE] [--epochs EPOCHS] [--eval-every EVAL_EVERY]
                         [--patience PATIENCE] [--disable-early-stopping]
                         [--deterministic | --non-deterministic] [-o]
                         [--allow-cross-commit-resume] [--force-resume] [--allow-config-mismatch]
                         [--workers WORKERS] [--batch-size BATCH_SIZE] [--grad-accum GRAD_ACCUM]
                         [--num-threads NUM_THREADS] [--no-cudnn-benchmark]
```

### Evaluation (`rmr_v3.eval`)
```text
python -m rmr_v3.eval --checkpoint CHECKPOINT [--manifest MANIFEST] [--output-dir OUTPUT_DIR]
                      [--device DEVICE] [--uniform-reliability] [--weighted-reliability]
                      [--tiling | --no-tiling] [--tta] [--dataset DATASET] [--split SPLIT]
                      [--config CONFIG] [--use-live-weights] [--eval-localization]
                      [--max-size MAX_SIZE] [--density-bins BIN_LO BIN_HI]
                      [--tile-blend {core,hann}]
```

---

## 3. Standalone CLI Commands (`configs/benchmarks/nextgen_sub65/`)

In compliance with repository rules (`AGENTS.md` Rule 2), zero batch automation scripts exist. Copy and execute individual commands directly. All configs enforce $\le 104,441$ trainable params, $0\%$ Knowledge Distillation, and unified hyperparameters (`crop_size: 512`, `init_m0: auto`, `bayesian_sigma: 8.0`, `radon_ot_scale_by_length: false`, `lr: 0.0003`, `min_lr_ratio: 0.05`, `ema_decay: 0.999`).

### A. Proposed Unified Cross-Dataset Suite (`SFILiteFPNNeck` + `DWSubpixelMeasureHead`, `100,596` params)

**1. ShanghaiTech Part A (`epochs: 1500`)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml
```
```bash
python -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml
```

**2. ShanghaiTech Part B (`epochs: 1500`)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_shb_synergistic_hdc_dw_subpixel_100k.yaml
```
```bash
python -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_shb_synergistic_hdc_dw_subpixel_100k.yaml
```

**3. UCF-QNRF (`epochs: 1000`)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_qnrf_synergistic_crop768_dw_subpixel_103k.yaml
```
```bash
python -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_qnrf_synergistic_crop768_dw_subpixel_103k.yaml
```

**4. JHU-Crowd++ (`epochs: 800`)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_jhu_synergistic_hdc_dw_subpixel_103k.yaml
```
```bash
python -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_jhu_synergistic_hdc_dw_subpixel_103k.yaml
```

### B. Single-Variable Ablation Suite on SHA (`epochs: 1500`)

**5. Ablation — `HDCLiteFPNNeck` Direct (`99,892` params)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_hdc_mcbpro_subpixel_103k.yaml
```

**6. Ablation — `ASPPLiteNeck` `[1,2,4]` Direct (`99,892` params)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_aspp124_dw_subpixel_100k.yaml
```

**7. Ablation — Learned `mcbpro` Solver w/ `SFILiteFPNNeck` (`103,226` params)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_synergistic_mcbpro_subpixel_103k.yaml
```

**8. Ablation — Canonical Stride-4 `ASPPLite+GAP+MCBPRO` (`104,390` params)**
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_canonical_stride4_mcbpro_repro_104k.yaml
```

---

## 4. Evaluation & Verification Commands

### Evaluate on Target Split + Localization
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.eval `
  --config configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml `
  --checkpoint runs_fixed/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k/best_val_mae.pt `
  --eval-localization
```

### Evaluate JHU-Crowd++ Official Test Split (`jhu_test.jsonl`)
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.eval `
  --config configs/benchmarks/nextgen_sub65/exp_jhu_synergistic_hdc_dw_subpixel_103k.yaml `
  --checkpoint runs_fixed/benchmarks/nextgen_sub65/exp_jhu_synergistic_hdc_dw_subpixel_103k/best_val_mae.pt `
  --manifest data/jhu_test.jsonl `
  --output-dir runs_fixed/benchmarks/nextgen_sub65/exp_jhu_synergistic_hdc_dw_subpixel_103k/eval_test
```

### Multi-Seed Verification (`seed 1337` / `2024`)
```powershell
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml --seed 1337 --output-dir runs_fixed/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k_s1337
.\.venv\Scripts\python.exe -m rmr_v3.trainer -c configs/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k.yaml --seed 2024 --output-dir runs_fixed/benchmarks/nextgen_sub65/exp_sha_direct_sfi_dw_subpixel_102k_s2024
```

### Core & Loss Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/core/test_cross_dataset_universality.py tests/core/test_non_aspp_necks.py tests/rmr_v3/test_radon_ot.py tests/rmr_v23/test_rmr_v23.py
```
