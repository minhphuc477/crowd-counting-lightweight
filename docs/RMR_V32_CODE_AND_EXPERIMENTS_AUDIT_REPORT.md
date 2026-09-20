# RMR-v32 Comprehensive Code & Experimental Designs Audit Report

**Date:** 2026-09-21  
**Status:** Audit Complete — Actionable Implementation Plan Created  
**Scope:** `rmr_core/`, `rmr_v3/`, `configs/rmr_v32/`, empirical data distributions, and scientific ablation completeness.

---

## 1. Executive Summary

A deep, adversarial audit of the RMR-v32 codebase and its experimental designs was conducted to ensure mathematical soundness, software engineering integrity, and publication readiness for top-tier computer vision venues (CVPR / TPAMI). 

Three structural code defects and one statistical parameter mismatch were identified and resolved in the architectural plan. Furthermore, the experimental design suite was expanded from **7 initial configs** to a **complete 12-experiment matrix** covering the primary hypotheses, the formal theoretical ablations (Dynamic Windowing, Radon-Nikodym adjoint, Morozov deadband, SNR reliability), and multi-dataset evaluation (ShanghaiTech Part B).

---

## 2. Code Audit Findings (Deep Inspection)

### Finding 1: Rigid MobileNet Truncation Lock-in
- **Location:** `rmr_core/backbones.py` (lines 60–74)
- **Defect:** 
  ```python
  selected_module_names = list(self.backbone.feature_info.module_name())
  last_module = selected_module_names[-1]
  if not last_module.startswith("blocks."):
      raise RuntimeError(f"Cannot safely truncate {model_name}: last selected feature is {last_module!r}")
  ```
  `MobileNetV4Backbone` unconditionally asserts that the last feature module starts with `"blocks."`. 
- **Impact:** Passing any non-MobileNet backbone (such as `convnext_femto` where module is `stages.2`, `resnet50` where module is `layer3`, or `swin_tiny` where module is `layers.2`) immediately triggers a fatal `RuntimeError`. This breaks the multi-backbone generalization claimed in the paper.
- **Solution:** Refactor into `TimmPyramidBackbone`. Only apply block truncation if `model_name.startswith("mobilenetv4")`. For other families, rely directly on `timm.create_model(..., features_only=True, out_indices=selected_indices)`.

---

### Finding 2: Hardcoded Parameter Ceiling in Architecture Constructor
- **Location:** `rmr_v3/model/architecture.py` (line 189)
- **Defect:**
  ```python
  total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
  if total_trainable > 105000:
      raise ValueError(f"Strict parameter ceiling exceeded: {total_trainable} > 105,000 parameters.")
  ```
- **Impact:** The 105,000 ceiling is hardcoded inside `RMRv3.__init__`. While essential for the edge regime (MobileNetV4), it prevents instantiating the model with medium or server-scale backbones (e.g., ConvNeXt-Femto with ~5.2M params or ResNet-50 with ~25M params) for the comparative Pareto frontier analysis in Section 4 of the paper.
- **Solution:** Make the parameter budget configurable via `RMRv3Config.max_trainable_params` (default: 105,000 for edge models; set to 0 or appropriate budget for macro backbones).

---

### Finding 3: Dense Loss Scaling Threshold Statistical Mismatch
- **Location:** `configs/rmr_v32/rmr_v32_h4_dense_loss_scaling.yaml` (lines 157–162)
- **Defect:** `dense_loss_thresh: 400.0` was chosen based on full-image count distributions (> 500 people). However, training is conducted on **512x512 random crops**.
- **Empirical Measurement on Real Training Crops:**
  Running exact crop extraction on `data/sha_a_train_all.jsonl` ($N=200$ samples):
  - **Min:** 8.0, **Max:** 1521.0
  - **Mean:** 270.4, **Median (p50):** 181.0
  - **75th percentile (p75):** 357.0
  - **90th percentile (p90):** 599.0
  - **Crops with count > 400:** only **21.5%**!
  - **Crops with count > 250:** **33.0%**!
- **Impact:** With `dense_loss_thresh = 400.0`, **78.5%** of all training crops receive zero scaling boost (`boost = 0.0`). Crops with 250–399 people (which suffer severe undercounting in dense clumps) receive no loss scaling at all.
- **Solution:** Calibrate `dense_loss_thresh: 250.0`, `dense_loss_norm: 250.0`, `dense_loss_max_boost: 1.5`. This smoothly scales the top 33% dense crops from 1.0x up to 2.5x without destabilizing moderate crops.

---

## 3. Experimental Design Audit: The Full 12-Experiment Suite

An A* paper requires not only testing positive hypotheses, but also isolating the individual contribution of every mathematical operator through rigorous ablations. The suite is organized into 4 tracks:

| # | Config Name | Track | Target / Hypothesis | Trainable Params | Role in Paper |
|---|---|---|---|---|---|
| 1 | `rmr_v32_step0_anchor.yaml` | Track A | Baseline: 100% v19 replication | 104,441 | Table 1 Anchor |
| 2 | `rmr_v32_control_no_solver.yaml` | Track A | Control: Pure feedforward $y_0$ ($T=0$) | 104,441 | Table 3: Solver Gain |
| 3 | `rmr_v32_h1_cpcm.yaml` | Track A | H1: Continuous Perspective Carrier Modulation | 104,753 | Table 2: Sub-50 Candidate |
| 4 | `rmr_v32_h2_floor_suppression.yaml` | Track A | H2: $C^1$ Smooth Floor Suppression ($\tau=0.008$) | 104,441 | Table 2: Noise Suppression |
| 5 | `rmr_v32_h3_composite.yaml` | Track A | H3: CPCM + Smooth Floor composite | 104,753 | Table 2: Sub-50 Main Model |
| 6 | `rmr_v32_h4_dense_loss_scaling.yaml` | Track A | H4: Calibrated Dense Loss Weighting (thresh=250) | 104,441 | Table 2: Anti-Saturation |
| 7 | `rmr_v32_h5_conservative_solver.yaml` | Track A | H5: Zero TV Diffusion ($\lambda_{\text{TV}}=0$) | 104,441 | Table 3: Peak Preservation |
| 8 | `rmr_v32_ablation_static_windowing.yaml` | Track B | Ablation: Static Windowing ($A_\pi^\top \to A_{\text{static}}^\top$) | 103,958 | Table 3: Proves Theorem 2 |
| 9 | `rmr_v32_ablation_flat_adjoint.yaml` | Track B | Ablation: Naive Back-Projection ($A_{\text{flat}}^\top$) | 104,441 | Table 3: Proves Claim 2 |
| 10 | `rmr_v32_ablation_no_morozov.yaml` | Track B | Ablation: Unconstrained Fitting ($\gamma = 0.0$) | 104,441 | Table 3: Proves Theorem 3 |
| 11 | `rmr_v32_ablation_uniform_reliability.yaml` | Track B | Ablation: Unweighted Fitting ($W = I$) | 104,441 | Table 3: Proves SNR Weights |
| 12 | `rmr_v32_shb_anchor.yaml` | Track C | Multi-Dataset: ShanghaiTech Part B Benchmark | 104,441 | Table 4: Cross-Dataset |

---

## 4. Verification Protocol
1. **Parameter Ceiling Compliance:** Every edge config strictly maintains trainable parameters $\le 105,000$.
2. **File Length Limits:** Every `.py` file strictly maintains line count $\le 450$.
3. **Mathematical Invariant:** Zero KaTeX errors, zero leaking batch-averaged scalings, zero ad-hoc splits.
