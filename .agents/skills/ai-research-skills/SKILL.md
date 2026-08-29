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
