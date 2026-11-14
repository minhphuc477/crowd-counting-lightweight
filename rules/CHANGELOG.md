# Changelog

All notable changes to the Constrained Radon Measure Recovery (`rmr_v3`) repository are documented here following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased / RMRv3 Unified Framework] - 2026-11-14

### Fixed
- **High-Frequency Background Clutter in `SFILiteFPNNeck` ([`rmr_core/necks/sfi.py`](file:///f:/lightweightcrcn/rmr_core/necks/sfi.py))**:
  - Co-gated the Laplacian high-frequency impulse enhancer with both semantic context gate `gate4` and parameter-free Local SimAM Neuro-Energy Attention (`_local_simam_gate`), keeping neck parameters strictly at `10,432` and total model parameters at `100,596` (0 additional parameters). This eliminates texture clutter leakage on sparse backgrounds while amplifying isolated head impulses in dense crowd regions.
- **`GroupNorm` Spatial Variance Compression & Dense Undercounting Bias**:
  - Enabled `density_adaptive_scale: true` with `density_scale_gamma: 0.25` across all polymorphic heads (`FineMeasureHead`, `ResidualDilatedFineHead`), scaling positive peak logits super-linearly ($y = y_{\text{base}} \cdot (1 + 0.25 \cdot \text{ReLU}(z))$) to counteract spatial variance compression on Dense scenes without distorting sparse/background cells.
- **Pretrained Backbone Weight Thrashing on Low-Count Benchmarks (SHB & JHU)**:
  - Adjusted `backbone_lr_scale: 0.1` and `lr: 0.0002` on SHB and JHU, stabilizing pretrained feature representations over extended training horizons (1500 and 800 epochs).
- **Double `0.25x` Density Initialization Bug in [`rmr_v3/engine.py`](file:///f:/lightweightcrcn/rmr_v3/engine.py)**:
  - Removed the redundant `0.25 *` multiplier inside `make_model()` when `init_m0` is `None` or `"auto"`, bringing `rmr_v3/engine.py` into exact alignment with [`rmr_v3/trainer.py`](file:///f:/lightweightcrcn/rmr_v3/trainer.py) and `FineMeasureHead` (which divides `m0` by `4.0` internally when `subpixel_stride2=True`).
- **Corrected False Peer-Reviewed Baseline Citation**:
  - Replaced the erroneous legacy claim attributing `EBC-ZIP-S` to `84k` params / `57.2` MAE in `input_contract.md` with verified 2025 EBC-ZIP scaling figures (`EBC-ZIP-P`: `0.81M`, SHA `71.18`; `EBC-ZIP-N`: `3.36M`, SHA `58.86`; `EBC-ZIP-T`: `10.53M`, SHA `56.36`; `EBC-ZIP-S`: `33.60M`, SHA `55.17`; `EBC-ZIP-B`: `105.60M`, SHA `47.81`) alongside verified sub-$200k$ standalone baselines (`TinyCount` 60k -> 78.2, `Lw-Count` 71k -> 69.7, `MCNN` 133k -> 110.2, `LSANet` 200k -> 66.1).

### Changed
- **Eliminated Dataset-Dependent Configuration Tuning (`configs/benchmarks/nextgen_sub65/`)**:
  - Unified all 4 primary cross-dataset configs (`sha`, `shb`, `qnrf`, `jhu`) under one identical architecture (`SFILiteFPNNeck` with `hdc_dilations: [1,2,3]`, `use_aspp_gap: false`, `DWSubpixelMeasureHead`, `output_stride: 2`, `subpixel_stride2: true`, `head_activation: softplus`, Direct mode, `100,596` trainable params, `0%` KD), identical `init_m0: auto`, identical data augmentations (`crop_size: 512`, `pad_small_images: true`, `max_size: 2048`, `scale_range: [0.6, 1.4]`), identical losses (`dual_bayesian_radon_ot` with `bayesian_sigma: 8.0`, `bayesian_sigma_max: 8.0`, `radon_ot_scale_by_length: false`, `count_nb_dispersion: 50.0`), and identical optimizer/EMA hyperparameters (`lr: 0.0003`, `backbone_lr_scale: 0.25`, `min_lr_ratio: 0.05`, `ema_decay: 0.999`, full benchmark horizons: `1500` SHA, `1500` SHB, `1000` QNRF, `800` JHU).
  - Standardized all 4 SHA single-variable ablation configs (`hdc_lite`, `aspp_lite`, `sfi_lite + mcbpro`, `canonical stride-4 mcbpro`) onto the exact same unified `data`/`loss`/`train` baseline (`epochs: 1500`, `min_lr_ratio: 0.05`), eliminating multi-variable confounders and restoring full Cosine Annealing convergence budgets.
- **Repository Documentation Overhaul & Legacy Cleanup**:
  - Removed 36 legacy/biased `.md` reports across `docs/archive/`, `docs/code_explanation/`, obsolete `docs/*`, and `rules/LUU_Y_THIET_KE_THI_NGHIEM_ANTI_REGRESSION.md`, plus all temporary `.md` artifacts.
  - Rewrote official repository docs around a 4-step survey-anchored research framework (`2024–2025` crowd counting surveys), strict schema-validated CLI entrypoints, verified `runs/` + `runs_fixed/` + `results/` metrics, and explicit failure-mode disclosures.

---

## [3.x Audited Benchmarks & Forensics History]

### Measured & Audited (On-Disk Artifacts)
- **Canonical Track A Stride-4 MCBPRO (`rmr_canonical_104k_seed42`, 104,390 params)**:
  - SHA Seed 42 Raw `65.62 MAE / 111.32 RMSE` (`TTA: 65.69 / 109.10`); Seed 7 Raw `72.20` (2-seed mean `68.91 +/- 4.65`).
  - Solver Isolation Probe: `FineHead y0` alone `65.23 / 108.40`, Full learned solver `65.57 / 111.34`, Oracle GT-box solver `4.77 / 5.78`.
- **182-Image Mass Decomposition Forensics ([`results/forensics/full_182_forensic_results.json`](file:///f:/lightweightcrcn/results/forensics/full_182_forensic_results.json))**:
  - Measured `sum_y0 = 81,629.66` vs `sum_gt = 78,862.0`, with `bg_mass_5 = 53,528.18` (`65.57%`) outside $5\times 5$ GT head neighborhoods and Softplus bias saturation near $b \approx -5.442$.
- **Audited `runs/benchmarks/` & `runs_fixed/benchmarks/`**:
  - Established single-seed and 3-seed baselines across SHA (`68.05` `runs_fixed` Direct Pretrained; `69.46 +/- 1.66` `runs` 3-seed), SHB (`9.24` `runs_fixed` Scratch Direct; `9.63` Pretrained Direct), UCF-QNRF (`103.35` `runs_fixed` Pretrained Direct; `102.10` `runs` 768-crop), JHU-Crowd++ Val (`58.07` `runs_fixed` Pretrained Direct; `76.58` Official Test), and UCF_CC_50 (5-fold CV: `298.65 +/- 85.18` Scratch, `329.99 +/- 100.63` Pretrained).
