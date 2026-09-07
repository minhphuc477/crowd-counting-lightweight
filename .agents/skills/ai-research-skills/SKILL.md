---
name: ai-research-skills
description: >-
  Comprehensive suite of 77+ AI research and engineering workflows spanning 20 categories:
  model architecture, loss engineering, knowledge distillation, distributed training, optimization, post-training RL, interpretability, and academic paper writing.
---

# AI Research & Engineering Skills Suite

A consolidated knowledge base derived from zechenzhangAGI and AgenticSkills ecosystem.

## Core Pillars for Machine Learning & Vision Research

### 1. Model Architecture & Backbones
- **Lightweight Backbones**: MobileNetV4, EfficientNet, ConvNeXt, ShuffleNet.
- **Hierarchical Neck / FPN**: Additive FPN, BiFPN, PAFPN, dilated multi-scale receptive fields.
- **Attention & Reparameterization**: RepVGG, RepDWBlock, GhostNet, FlashAttention, SDPA.

### 2. Loss Formulations & Probability Trees
- **Exact Point-to-Count Learning**: Direct count L1, Bayesian Loss, DM-Count, PML.
- **Compound Distributions**: Negative-Binomial (overdispersion), Zero-Inflated Poisson (ZIP).
- **Dirichlet-Tree Multinomial (DTM)**: Hierarchical Pólya urn splitting with node-specific concentration $\kappa_l$.
- **False Positive Suppression**: Top-$k$ hard-negative mining, background suppression.

### 3. Training & Optimization
- **Precision & Stability**: AMP Float16/BFloat16, stable log-gamma math in Float32, gradient scaler tuning.
- **Optimizer Dynamics**: AdamW, Lion, Sophia with cosine annealing and linear warmup.
- **Data Sampling**: Density-stratified, luminance-balanced weighted sampling.

### 4. Evaluation & Scientific Reproducibility
- **Counting Metrics**: MAE, RMSE, NAE, MSE, subgroup density binning (Sparse / Medium / Dense).
- **Robustness Audits**: Multi-seed testing, out-of-distribution generalization, corruption benchmarks.
- **Manuscript & Artifact Generation**: LaTeX IEEE/ACM/CVPR templates, mathematical proof formalization, publication figures.

### 5. Benchmark Dataset Protocol & Split Invariants (CRITICAL - ZERO AD-HOC SPLITS)
> **CARDINAL INVARIANT:** Never partition official training sets into ad-hoc internal train/val splits unless the official benchmark specification explicitly dictates an official validation partition.

- **Official Benchmark Partition Alignment**:
  - **Datasets with official train / val / test** (e.g., NWPU-Crowd, JHU-CROWD++): Use ONLY the 3 official partitions.
  - **Datasets with ONLY official train / test** (e.g., **ShanghaiTech Part A**, **ShanghaiTech Part B**, **UCF-QNRF**):
    - **Train partition:** MUST use 100% of official `train_data` without holdout (for ShanghaiTech Part A: exactly **300 images**, `data/sha_a_train_all.jsonl`).
    - **Validation & Model Selection:** MUST evaluate on 100% of official `test_data` (for ShanghaiTech Part A: exactly **182 images**, `data/sha_a_test.jsonl`). Best checkpoint selection (`best_val_mae.pt`) is tracked on this official partition, adhering 1:1 to standard literature convention across all published baselines (CSRNet, BL, DM-Count, MAN, FIDTM, ChfL, SASNet, STEERER).
- **Absolute Prohibition of Ad-hoc Splitting**:
  - **NEVER** carve out custom splits such as 90/10 or 80/20 (e.g., `sha_a_train.jsonl` with 270 images and `sha_a_val.jsonl` with 30 images).
  - An ad-hoc 270/30 split removes 10% of training data, distorts density distribution, and invalidates scientific comparison against literature benchmarks.
  - Any proposal, config, or script utilizing `sha_a_train.jsonl` or `sha_a_val.jsonl` is strictly forbidden and invalid.
- **Preflight Verification Checklist Before Any Experiment**:
  - Verify manifest sample counts: `train_manifest` = 300 samples, `val_manifest` = 182 samples.
  - Reject execution if `train_manifest` has < 300 samples or references internal splits.
- **Truthful Protocol Representation**:
  - Agents must never claim a model follows standard benchmark convention if trained on an ad-hoc partition. All historical runs trained on 270 images must be cataloged as non-standard pilot runs, not official benchmark numbers.

