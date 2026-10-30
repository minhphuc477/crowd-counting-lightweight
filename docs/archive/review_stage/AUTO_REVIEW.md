# AUTO REVIEW: Autonomous Research Improvement Loop

## Round 1 — Initial Research Review & Architectural Audit

**Date**: 2026-10-20  
**Target Venue**: CVPR / ICCV / ECCV / TPAMI  
**Evaluation Benchmark**: ShanghaiTech Part A Canonical (300 Train / 182 Test)  
**Strict Invariants**:
- Parameters: $\le 104,441$ (Track A) / $\le 200,000$ (Track B)
- Standalone: Exactly $0.0\%$ Knowledge Distillation
- Optimizer: 1,000 epochs, Cosine Annealing, no WSD, no premature decay at epoch 650

---

### Scorecard
- **Score**: 5.5 / 10
- **Verdict**: `not ready`
- **Primary Bottleneck**: Gap between Sparse/Moderate MAE (10–53 MAE) and Dense MAE (122–161 MAE) across 46 dense test images.

---

### Critical Weaknesses Identified (Ranked by Severity)

1. **Weakness 1 (Mathematical): SIRT Multiplicative Update Zero-Support Trap**
   - *Symptom*: When $y_0(u) = 0$, $\Delta y(u) = 0 \cdot A^T r = 0$. The solver cannot inject mass into cells where the carrier initially predicted 0.
   - *Minimum Fix*: Transition from iterative SIRT to the **Closed-Form Normalized Mass Transport Operator**:
     $$y^*(u) = P(u) \sum_{k=1}^K \pi_k(u) \frac{b_k}{\max\left(\int_{R_k} P(v) \, dv, \, \epsilon\right)}$$
     or Additive-Multiplicative Forward-Backward Splitting.

2. **Weakness 2 (Loss Gradient): Negative-Binomial Heteroscedastic Gradient Attenuation**
   - *Symptom*: $\frac{\partial \mathcal{L}_{\text{NB}}}{\partial \mu} = \frac{\mu - b}{\sigma^2(\mu)}$ collapses to $O(1/\mu)$ in dense boxes with $\mu = 100-300$, resulting in regression slope $0.23\times$.
   - *Minimum Fix*: Implement **Beta-NLL** ($\beta = 1.0$) with variance-detached weighting:
     $$\mathcal{L}_{\beta\text{-NB}} = \text{stop\_gradient}(\sigma^2) \cdot \mathcal{L}_{\text{NB}}$$
     restoring a constant $1.0\times$ linear gradient slope $(\mu - b)$ across all density tiers.

3. **Weakness 3 (Spatial Resolution): Spatial Aliasing at Stride 4**
   - *Symptom*: 38.3% of head pairs in dense images fall into the same $4 \times 4$ pixel cell, saturating scalar density predictions.
   - *Minimum Fix*: Deploy **Subpixel Stride 2 via Zero-Parameter PixelShuffle** ($r=2$), reducing cell width to $2 \times 2$ pixels and quadrupling spatial resolution from $128 \times 128$ to $256 \times 256$ with exactly 0 extra parameters.

4. **Weakness 4 (Optimization): Auxiliary Loss Conflict**
   - *Symptom*: 8 active loss terms pulling representations in competing directions.
   - *Minimum Fix*: Purify to a minimal 3-loss orthogonal suite:
     1. $\mathcal{L}_{\text{point}}$: Peak localization loss on $P(u)$ (Focal / FIDT Peak L1).
     2. $\mathcal{L}_{\text{region}}$: Beta-NB count loss on regional boxes $b_k$.
     3. $\mathcal{L}_{\text{global}}$: Exact global count conservation loss $|\sum_u y^*(u) - N_{\text{gt}}|$.

---

### Action Items for Round 2
1. Verify mathematical proofs and derivations in `DERIVATION_PACKAGE.md`.
2. Implement Closed-Form Mass Transport Operator in `rmr_v3/model/architecture.py` and `rmr_v3/model/solver_step.py`.
3. Implement Beta-NLL loss with detached variance weighting in `rmr_v3/losses/regional.py`.
4. Run adversarial preflight checks: batch sample isolation, odd resolution invariance, and numerical stability.
