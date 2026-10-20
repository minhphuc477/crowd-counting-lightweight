# Reviewer Memory: Autonomous Research Review Loop

## Round 1 — Baseline Pre-Audit Memory

### Context & Stated Objective
- **Goal**: Reach ~50 MAE on ShanghaiTech Part A canonical benchmark (300 train / 182 test images) under strictly $\le 104,441$ parameters (Track A) or $\le 200,000$ parameters (Track B).
- **Core Stance**: Standalone 100% (0.0% Knowledge Distillation). No teacher models, no fake oracles.

### Reviewer Suspicions & Tracked Flaws
1. **Zero-Support Multiplicative Trap**:
   - The unrolled SIRT solver update $\Delta y_t = \omega_t \frac{y_t}{\text{cov}} A^T r_t$ cannot generate mass where $y_0 = 0$. In dense clusters where the background bias ($b_0 \approx -4.14$) suppresses initial logits, heads are permanently missed.
2. **Heteroscedastic Gradient Attenuation on Dense Boxes**:
   - The Negative-Binomial loss derivative $\frac{\partial \mathcal{L}_{\text{NB}}}{\partial \mu} = \frac{\mu - b}{\mu + \mu^2 / r}$ scales as $O(1/\mu^2)$, dampening gradients on 128px dense boxes by over $1,000\times$, resulting in an empirical slope of only $0.23\times$.
3. **Loss Function Sprawl & Gradient Interference**:
   - Stacking 6–8 auxiliary losses (hurdle, trunc_nb, curvature, scale_align) caused gradient cancellation ($\cos \theta < 0$).
4. **Resolution Bottleneck at Stride 4**:
   - 38.3% of head pairs in dense crops are $< 4\text{px}$ apart, falling into the same $4 \times 4$ pixel cell. Subpixel Stride 2 is physically necessary to separate heads without increasing parameter count.
