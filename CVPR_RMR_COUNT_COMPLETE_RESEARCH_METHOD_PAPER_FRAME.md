# Regional Measure Reconciliation for Ultra-Light Crowd Counting
## Literature-grounded research framing, Research Questions, method, novelty boundary, and CVPR-style paper plan

**Working method name:** RMR-Count  
**Status:** research specification / paper blueprint — not a finished claim of novelty or SOTA  
**Project:** light crowd counting  
**Date:** 2026-09-05  

> **Central discipline of this document:** the paper must start from a limitation that already exists in the literature, derive a gap that survives prior art, formulate falsifiable Research Questions, and only then introduce the method as an answer.  
> The method must **not** invent its own problem after the architecture has already been chosen.

---

# 0. Executive decision

The clean paper is **not**:

> “We invented a cumulative operator and then found a reason to use it.”

It is also **not**:

> “Existing crowd counters lack local consistency.”

That claim is false as a novelty basis: local-count objectives, grid losses, composition losses, regional consistency, divide-and-conquer, and even losses controlling errors over all sub-regions already exist.

The defensible research problem is the intersection of three independently established observations:

1. **Crowd counting is dominated by fine/local visual evidence.**  
   Human heads are often very small; recent work explicitly shows that excessive receptive-field expansion can add computation without proportionate benefit, and that local modeling is a strong design principle for crowd counting.

2. **Regional count structure is useful, but prior work usually uses it as a target, loss, regularizer, aggregation rule, or multi-scale prediction constraint.**  
   Local Counting Map (LCM), grid/local-count losses, Composition Loss, relative local counting, characteristic-function losses, divide-and-conquer, and related methods already exploit region-level counts.

3. **Ultra-light models have very little capacity to spend on generic contextual modeling.**  
   Lightweight crowd counters repeatedly trade accuracy against the cost of multi-scale branches, context modules, attention, large receptive fields, or deeper backbones.

The surviving gap candidate is therefore:

> **Existing crowd-counting work extensively exploits regional counts for supervision, aggregation, or hierarchical prediction, but we did not find a method that exposes overlapping regional counts as an explicit inference-time measurement space and uses the exact adjoint of regional summation to convert regional count disagreements into corrections of a fine non-negative counting measure.**

This gap is narrower than “regional consistency,” but it is scientifically cleaner.

The paper asks whether an ultra-light counter can use **known counting algebra** for part of its regional reasoning instead of learning all regional interaction through additional neural capacity.

---

# 1. Evidence hierarchy used in this document

To avoid mixing established facts with our own proposal, every major statement belongs to one of three classes.

### [L] Literature-supported fact
A statement explicitly supported by prior published work.

### [I] Inference / synthesis
A conclusion drawn by comparing multiple papers. It is plausible but is not a quotation or explicit claim from one paper.

### [P] Proposed hypothesis or method
A new hypothesis, design, or planned experiment from this project.

The paper should follow the same discipline.

---

# 2. Literature-grounded limitations

## 2.1 Limitation L1 — crowd counting needs strong local evidence

### Evidence

**Local Information Matters: A Rethink of Crowd Counting**, Pan and Jia, ECAI 2025, argues from data and effective-receptive-field analysis that crowd counting differs from classification because individual heads typically occupy a small part of the image. The paper reports an average JHU-CROWD++ head size of 16.2 pixels and that 95.5% of heads are smaller than 50 pixels. It further argues that simply enlarging theoretical receptive field can increase computation while bringing limited benefit for the small individuals that dominate the task.

This does **not** imply that global context is useless. The same paper adds a small global mechanism for rare large heads. The relevant conclusion is more precise:

> **[L] Fine local modeling is unusually important in crowd counting, and generic pursuit of ever-larger learned receptive fields is not automatically the best use of compute.**

Reference:
- Tianhang Pan, Xiuyi Jia, *Local Information Matters: A Rethink of Crowd Counting*, ECAI 2025, DOI: 10.3233/FAIA250799.

---

## 2.2 Limitation L2 — contextual and multi-scale modeling is useful but consumes capacity

Crowd counting has long used multi-scale columns, dilation, pyramids, attention, mixture-of-experts, and context modules to handle perspective and density variation.

**FusionCount** explicitly identifies an efficiency issue: many encoder-decoder counters underuse earlier encoder features and add separate multi-scale extraction modules, which increase computational cost. FusionCount tries to recover multi-scale information by reusing encoded features instead of adding more extraction modules.

**LRMBNet**, **TinyCount**, **LCDnet**, and related lightweight methods likewise exist because the accuracy of heavier context-rich networks is difficult to retain under deployment constraints.

The relevant conclusion is:

> **[L] There is a real accuracy–context–efficiency trade-off in lightweight crowd counting.**

This is not a new problem invented by this project.

References:
- Yiming Ma, Víctor Sánchez, Tanaya Guha, *FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion*, ICIP 2022.
- Mingze Li, Diwen Zheng, Shuhua Lu, *Lightweight Res-Connection Multi-Branch Network for Highly Accurate Crowd Counting and Localization*, CMC 2024.
- Hyeonbeen Lee, Jangho Lee, *TinyCount: An Efficient Crowd Counting Network for Intelligent Surveillance*, Journal of Real-Time Image Processing, 2024.
- Muhammad Asif Khan et al., *LCDnet: A Lightweight Crowd Density Estimation Model for Real-Time Video Surveillance*, Journal of Real-Time Image Processing, 2023.

---

# 3. Regional counting is already prior art

A clean paper must acknowledge this aggressively.

## 3.1 Density maps are already measures whose regional integral gives counts

The classical density-map paradigm already satisfies:

\[
N(R)=\int_R d(x)\,dx.
\]

Therefore:

> **“We can count arbitrary regions by integrating a map” is not novel.**

---

## 3.2 Local Counting Map already moves supervision toward regional counts

Liu et al., ECCV 2020, introduce **Local Counting Map (LCM)**. Each LCM value represents a local patch count rather than a pixel density. They establish:

\[
\mathrm{MAE}\le \mathrm{LCME}\le \mathrm{DME},
\]

and argue that local-count supervision better aligns training with counting evaluation than purely pixel-wise density-map error.

Therefore:

> **“Local counts are a better supervision target” is not our gap.**

Reference:
- Xiyang Liu, Jie Yang, Wenrui Ding, *Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting*, ECCV 2020.

---

## 3.3 Local inconsistency is already a known problem

*Towards Locally Consistent Object Counting with Constrained Multi-stage CNNs* explicitly observes that an image may have a nearly correct global count while containing large sub-region errors and background false mass. It introduces multi-stage refinement and a grid loss to improve local consistency.

Therefore:

> **“Global count can hide local errors” is not our novel observation.**

---

## 3.4 Regional/local-count losses are already crowded

Prior examples include:

- MESA-style region constraints;
- local count loss in CRDNet;
- grid loss for locally consistent counting;
- LCM;
- relative local counting in HMoDE;
- self-calibrated region losses;
- Composition Loss;
- hierarchical / cross-scale consistency losses.

Therefore:

> **“Use regional count error in the loss” is not our contribution.**

---

## 3.5 Frequency-domain supervision already controls all sub-regions theoretically

Shu et al., CVPR 2022, formulate crowd counting in the frequency domain using characteristic functions of finite measures. Their characteristic-function loss is proven to upper-bound a pseudo-sup metric over normalized counting errors of all spatial sub-regions.

The later GCFL formulation extends this framework.

Therefore:

> **We cannot claim that previous work fails to supervise regional counts comprehensively.**

References:
- Weibo Shu et al., *Crowd Counting in the Frequency Domain*, CVPR 2022.
- Weibo Shu, Jia Wan, Antoni B. Chan, *Generalized Characteristic Function Loss for Crowd Analysis in the Frequency Domain*, TPAMI 2024.

---

## 3.6 Region hierarchy at inference is also not new

S-DCNet / SS-DCNet use spatial divide-and-conquer to predict and combine local counts at multiple resolutions.

*Divide and Count* uses local image divisions and inclusion-exclusion for overlapping regions.

Therefore:

> **“Use local regions at inference” or “use inclusion-exclusion” is not new.**

References:
- Haipeng Xiong et al., *From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer*, ICCV 2019.
- Silvia Laura Pintea et al., *Divide and Count: Generic Object Counting by Image Divisions*, IEEE TIP.

---

# 4. What is actually missing?

After removing the prior-art claims above, a narrower question remains.

Most regional-count approaches fall primarily into one of these categories:

### Category A — regional information as supervision

\[
\theta^*
=
\arg\min_\theta
L_{\text{pixel}}
+
\lambda L_{\text{region}}.
\]

Regional structure affects **training gradients**, but once training is finished, inference is still approximately:

\[
I\rightarrow f_\theta(I)\rightarrow Y.
\]

Examples include LCM-related objectives, grid/local-count losses, relative local counting, Composition Loss constraints, and characteristic-function losses.

---

### Category B — regional predictions as outputs that are fused or selected

Examples include:

- local count regression;
- divide-and-conquer;
- hierarchical local counters;
- mixture-of-expert prediction;
- inclusion-exclusion aggregation.

These methods make regional predictions at inference, but they do not formulate the disagreement between a fine count measure and overlapping regional measurements as a residual in a measurement space whose **exact adjoint** updates the fine measure.

---

### Category C — generic learned iterative refinement

Examples include:

- Iterative Crowd Counting;
- recurrent spatial-aware refinement;
- cascaded residual density refinement;
- uncertainty-guided residual refinement.

A generic form is:

\[
Y^{(t+1)}
=
Y^{(t)}
+
R_\theta(F,Y^{(t)}).
\]

The neural network learns both:

1. **what is inconsistent**, and
2. **how that inconsistency should be mapped back spatially**.

---

# 5. Verified gap candidate

The gap should be stated cautiously:

> **[I] Regional count information is well established as supervision and as a prediction/aggregation device, while iterative refinement is also established. However, in the literature reviewed for this project, we did not find a crowd-counting method that treats overlapping regional counts as an explicit inference-time measurement space and uses the exact adjoint of regional summation to back-project regional count residuals into a fine non-negative count measure.**

This is the gap to test.

It is **not yet a “first-ever” claim**.

A final submission must still perform an updated collision search.

---

# 6. Why this gap matters specifically for ultra-light counting

The point is not that model-based inference is universally superior.

The reasoning is:

1. **[L]** small heads require fine visual evidence;
2. **[L]** multi-scale/global context can improve counting but costs model capacity;
3. **[L]** regional counts are mathematically additive and have been useful as supervision;
4. **[I]** ultra-light models may be wasting scarce parameters learning regional aggregation/correction structure that is partly known exactly from counting algebra.

This yields the central scientific question:

> **Can exact counting operators reduce the amount of learned contextual machinery needed by an ultra-light crowd counter?**

That is a field-motivated question, not a model-invented one.

---

# 7. Research Questions

A clean CVPR paper should use **two primary RQs**.

## RQ1 — operator versus learned context

> **RQ1. Under a fixed ultra-light parameter and compute budget, can explicit regional counting operators provide useful multi-scale count context more effectively than allocating the same budget to additional learned contextual/refinement layers?**

### Falsifiable hypothesis H1

\[
\boxed{
\text{RMR}
>
\text{same-cost learned context/refinement}
}
\]

in the accuracy–efficiency Pareto sense.

This must be tested with matched parameter/MAC controls.

---

## RQ2 — inference-time regional reasoning versus training-only regional supervision

> **RQ2. Does regional count information provide additional value when it is used to reconcile the current prediction at inference time, beyond using the same regional information only as a training loss or auxiliary prediction target?**

### Falsifiable hypothesis H2

Let:

- \(B\): direct fine-measure model;
- \(B+L_R\): same model with regional-count supervision only;
- \(B+H_R\): regional count head but no reconciliation;
- \(B+\mathrm{RMR}\): regional head plus inference-time reconciliation.

The hypothesis is:

\[
\boxed{
\mathrm{Err}(B+\mathrm{RMR})
<
\mathrm{Err}(B+L_R)
}
\]

and

\[
\boxed{
\mathrm{Err}(B+\mathrm{RMR})
<
\mathrm{Err}(B+H_R).
}
\]

If this does not hold, the central inference claim fails.

---

## Mechanistic sub-question

A reviewer will naturally ask:

> Is the exact adjoint actually useful, or is any additional refinement block sufficient?

This is not a third headline RQ. It is a mechanism test under RQ2:

\[
\boxed{
A^\top\text{-based reconciliation}
\quad
\text{vs.}
\quad
\text{same-cost generic CNN refinement}.
}
\]

---

# 8. Proposed answer: Regional Measure Reconciliation (RMR)

The method should be presented as **one mechanism**, not a catalogue of modules.

The method has only three conceptual pieces:

1. a lightweight local visual estimator;
2. a regional evidence estimator;
3. an exact regional measurement/reconciliation operator.

Everything else is implementation support.

---

# 9. Problem formulation

Given image:

\[
I\in\mathbb R^{3\times H\times W},
\]

and point annotations:

\[
\mathcal P=\{p_n\}_{n=1}^{N},
\]

define a non-negative discrete counting measure on an output grid:

\[
Y\in\mathbb R_+^{H_g\times W_g}.
\]

For stride \(s\):

\[
Y^{gt}_{ij}
=
\#\left\{
p_n:
\left\lfloor
\frac{y_n+0.5}{s}
\right\rfloor=i,\,
\left\lfloor
\frac{x_n+0.5}{s}
\right\rfloor=j
\right\}.
\]

The image count is:

\[
N=\sum_{ij}Y_{ij}.
\]

The canonical proposed output stride is:

\[
\boxed{s=4}.
\]

This is chosen to preserve local evidence; it is not claimed as novel.

---

# 10. Local observation network

Use a native lightweight encoder.

A first canonical configuration:

| Stage | Stride | Channels | Blocks |
|---|---:|---:|---:|
| Stem | 2 | 16 | 1 |
| \(C_4\) | 4 | 24 | 2 |
| \(C_8\) | 8 | 40 | 3 |
| \(C_{16}\) | 16 | 64 | 2 |

Block:

```text
DWConv 3×3
→ normalization
→ 1×1 expansion
→ SiLU
→ 1×1 projection
→ residual
```

Fuse:

\[
F
=
\phi(
P_4
+
U_2(P_8)
+
U_4(P_{16})
),
\]

where all projected tensors have width 32.

No transformer, MoE, ASPP, or large global attention is required in the core method.

This encoder is an **enabling component**, not a novelty claim.

---

# 11. Initial fine count measure

The local head predicts logits:

\[
z^{(0)}
=
H_Y(F).
\]

The initial count measure is:

\[
\boxed{
Y^{(0)}
=
\operatorname{softplus}(z^{(0)}).
}
\]

Therefore:

\[
Y^{(0)}\ge0.
\]

The initial count is:

\[
\hat N^{(0)}
=
\sum Y^{(0)}.
\]

This branch tests whether local visual evidence alone is sufficient.

---

# 12. Regional measurement space

Let:

\[
\mathcal R=\{R_m\}_{m=1}^{M}
\]

be a set of overlapping image-grid rectangles.

Define:

\[
A:
\mathbb R^{H_gW_g}
\rightarrow
\mathbb R^M
\]

such that:

\[
\boxed{
(AY)_m
=
\sum_{p\in R_m}Y_p.
}
\]

Thus \(AY\) contains the regional counts implied by the fine measure.

This operator is **not learned**.

---

# 13. Cumulative implementation of the forward operator

Define:

\[
C=P(Y),
\]

where:

\[
C_{ij}
=
\sum_{a\le i,b\le j}Y_{ab}.
\]

For rectangle:

\[
R=[y_1,y_2)\times[x_1,x_2),
\]

the count is:

\[
Q_R(C)
=
\tilde C(y_2,x_2)
-
\tilde C(y_1,x_2)
-
\tilde C(y_2,x_1)
+
\tilde C(y_1,x_1).
\]

Therefore:

\[
AY
=
Q(PY).
\]

The cumulative table is an **efficient implementation** of the measurement operator.

We do **not** claim prefix sums, integral images, or four-corner inclusion-exclusion as novel.

---

# 14. Regional visual evidence

A fine count map and a regional visual descriptor have different inductive biases.

The fine branch estimates:

\[
Y^{(0)}
\]

from local aligned features.

The regional branch independently estimates:

\[
b_m
\approx
N(R_m).
\]

To avoid processing every region with a separate CNN, compute an integral feature table:

\[
S_F
=
P(F).
\]

For each region:

\[
\bar F_m
=
\frac{Q_{R_m}(S_F)}
{|R_m|}.
\]

A shared lightweight head predicts:

\[
\boxed{
b_m
=
\operatorname{softplus}
\left(
h_b(\bar F_m,g_m)
\right),
}
\]

where \(g_m\) contains normalized region geometry such as width, height, and area.

The head is shared across region scales.

### Optional reliability

A second scalar:

\[
s_m=h_s(\bar F_m,g_m)
\]

may model regional uncertainty:

\[
\omega_m
=
\exp(-s_m).
\]

However, uncertainty is **not part of the core novelty**.

The first decisive implementation should test both:

- fixed scale weights;
- predicted regional reliability.

If uncertainty does not materially improve the core model, remove it from the main paper.

---

# 15. Regional reconciliation energy

The current fine prediction implies regional counts:

\[
q^{(t)}
=
AY^{(t)}.
\]

The regional visual branch predicts:

\[
b.
\]

Define the regional consistency energy:

\[
\boxed{
E(Y)
=
\frac12
\left\|
W^{1/2}
(AY-b)
\right\|_2^2.
}
\]

Its exact gradient is:

\[
\boxed{
\nabla_Y E
=
A^\top W(AY-b).
}
\]

This is the key mathematical object in the method.

The network does not need to learn the geometry of mapping every regional error back to all participating cells.

That mapping is given by:

\[
A^\top.
\]

---

# 16. Exact adjoint back-projection

For a vector of regional residuals:

\[
e=AY-b,
\]

the adjoint:

\[
A^\top e
\]

adds each regional residual to all cells participating in that region.

A naive implementation is expensive, but rectangles permit a difference-array implementation.

For region:

\[
R=[y_1,y_2)\times[x_1,x_2)
\]

with scalar \(\alpha\), update a corner buffer \(D\):

\[
D[y_1,x_1] += \alpha,
\]

\[
D[y_1,x_2] -= \alpha,
\]

\[
D[y_2,x_1] -= \alpha,
\]

\[
D[y_2,x_2] += \alpha.
\]

Then:

\[
\boxed{
A^\top \alpha
=
P(D).
}
\]

Thus both:

\[
A
\]

and:

\[
A^\top
\]

have cumulative implementations.

---

# 17. Coverage normalization

Overlapping windows create unequal region coverage.

Define:

\[
c
=
A^\top \omega.
\]

The normalized residual field is:

\[
\boxed{
r^{(t)}
=
\frac{
A^\top
[
\omega\odot
(AY^{(t)}-b)
]
}{
A^\top\omega+\epsilon
}.
}
\]

This prevents central cells from receiving larger corrections solely because they belong to more windows.

---

# 18. Learned local preconditioning

The exact adjoint says **where a regional error applies**, but it does not know which cells inside a region visually contain people.

A very small local network predicts a positive diagonal preconditioner:

\[
M^{(t)}
=
M_\theta(
F,
Y^{(t)},
r^{(t)}
).
\]

For stability:

\[
M^{(t)}
=
m_{\min}
+
(m_{\max}-m_{\min})
\sigma(h_\theta).
\]

This network does **not** replace the mathematical residual.

It only controls local allocation/magnitude.

This is the division of labor:

### Exact counting algebra
- regional summation;
- overlap/additivity;
- residual geometry;
- back-projection support.

### Neural network
- appearance;
- scale cues;
- foreground/background;
- local allocation of correction.

---

# 19. Positive reconciliation update

Parameterize:

\[
Y^{(t)}
=
\operatorname{softplus}(z^{(t)}).
\]

Since:

\[
\frac{\partial Y}{\partial z}
=
\sigma(z),
\]

a gradient-like update in latent space is:

\[
\boxed{
z^{(t+1)}
=
z^{(t)}
-
\eta_t
M^{(t)}
\odot
\sigma(z^{(t)})
\odot
r^{(t)}.
}
\]

Then:

\[
\boxed{
Y^{(t+1)}
=
\operatorname{softplus}(z^{(t+1)}).
}
\]

Default:

\[
T=2.
\]

The final result is:

\[
Y^*=Y^{(T)}.
\]

Global count:

\[
\boxed{
\hat N
=
\sum Y^*.
}
\]

---

# 20. Why the update is not just an arbitrary recurrent CNN

Generic refinement:

\[
Y^{(t+1)}
=
Y^{(t)}
+
R_\theta(F,Y^{(t)}).
\]

RMR:

\[
Y^{(t+1)}
=
\mathcal U_\theta
\left(
Y^{(t)},
\underbrace{
A^\top W(AY^{(t)}-b)
}_{\text{analytically defined residual direction}}
\right).
\]

The learned part does not invent the full correction geometry.

This is the exact distinction that must be experimentally tested.

---

# 21. Generic deep unrolling is not our novelty

Learned Primal-Dual and related inverse-problem methods already incorporate a known forward operator and its adjoint/back-projection into learned iterations.

Therefore:

> **We must not claim that unrolling \(A/A^\top\) inside a neural network is generically new.**

Our provisional task-specific novelty is:

> **formulating crowd counting itself as reconciliation between a fine non-negative count measure and independently predicted overlapping regional count measurements, where the count-specific regional-sum operator and its exact adjoint define the inference residual geometry.**

This distinction must appear in Related Work.

Reference:
- Jonas Adler, Ozan Öktem, *Learned Primal-Dual Reconstruction*, IEEE TMI 2018 / arXiv 1707.06474.

---

# 22. Summed-area tables are also not our novelty

Neural networks have already used integral images / summed-area tables to obtain large receptive fields efficiently.

Examples:

- Burkov and Lempitsky, *Deep Neural Networks with Box Convolutions*, NeurIPS 2018.
- Zhang, Halber, Rusinkiewicz, *Accelerating Large-Kernel Convolution Using Summed-Area Tables*, 2019.

Therefore the paper may claim:

> “The rectangular counting operator admits an efficient cumulative implementation.”

It must **not** claim:

> “We introduce integral images into neural networks.”

---

# 23. Region family

The first canonical design should be simple.

At output stride 4, use rectangular scales corresponding approximately to:

\[
\boxed{
16,\ 32,\ 64,\ 128\text{ image pixels}
}
\]

plus:

- full-image count;
- optional 2×2 global partition.

Do not initially add arbitrary region proposals, learned boxes, or adaptive zoom.

Use overlapping centered windows.

The region set should be ablated by scale.

---

# 24. Training objective

The paper should avoid inventing another complicated loss.

The base visual counter must use the **same primary supervision** in all matched controls.

Recommended strategy:

\[
\boxed{
L
=
L_{\text{base}}
+
\lambda_R L_{\text{region}}.
}
\]

Where:

- \(L_{\text{base}}\) is a standard count-map / point-supervised loss shared by all compared variants;
- \(L_{\text{region}}\) trains the regional evidence branch.

Two acceptable implementation paths:

### Path A — clean first pilot

Use exact stride-4 cell counts and a balanced robust loss:

\[
L_{\text{base}}
=
\frac12
\mathbb E_{Y^{gt}>0}
\rho(Y^*-Y^{gt})
+
\frac12
\mathbb E_{Y^{gt}=0}
\rho(Y^*-Y^{gt}).
\]

### Path B — stronger final paper carrier

Use a strong established point-supervised loss such as PML or a matched DM-Count-style objective for **all** direct and RMR variants.

This is preferable for the final SOTA experiment if it materially raises the carrier ceiling, because the paper contribution is architecture/inference, not a new supervision loss.

---

## 24.1 Regional evidence loss

For exact region count:

\[
N_R
=
\#\{p_n\in R\},
\]

use:

\[
L_{\text{region}}
=
\frac1M
\sum_R
\rho(b_R-N_R).
\]

If heteroscedastic confidence is retained:

\[
L_{\text{region}}
=
\frac1M
\sum_R
\left[
e^{-s_R}
\rho(b_R-N_R)
+
\alpha s_R
\right].
\]

---

# 25. Important architectural simplification

The previous UL-CMR sketch included:

- cumulative-conditioned scale routing;
- count signatures;
- adaptive iterations;
- uncertainty;
- several optional residual modules.

That is too easy to turn into a “module catalogue.”

The first CVPR paper should **not** include those by default.

Core paper architecture:

```text
Image
  │
  ▼
Ultra-light local encoder
  │
  ▼
Stride-4 feature F
  │
  ├──────────────► Fine measure head ──► Y^(0)
  │
  └──────────────► Shared regional head ──► b_R

Y^(0) ──► exact A ──► regional implied counts
                       │
                       ▼
                  AY^(t) - b
                       │
                       ▼
                  exact A^T
                       │
                       ▼
             local preconditioner
                       │
                       ▼
                    Y^(t+1)
                       │
                    repeat
                       │
                       ▼
                      Y*
```

One central mechanism.

---

# 26. Full canonical architecture table

For a \(256\times256\) training crop:

| Component | Output | Shape |
|---|---|---|
| input | RGB | \(3\times256\times256\) |
| stem | local feature | \(16\times128\times128\) |
| stage 4 | \(C_4\) | \(24\times64\times64\) |
| stage 8 | \(C_8\) | \(40\times32\times32\) |
| stage 16 | \(C_{16}\) | \(64\times16\times16\) |
| fusion | \(F\) | \(32\times64\times64\) |
| fine head | \(Y^{(0)}\) | \(1\times64\times64\) |
| feature prefix | \(P(F)\) | \(32\times64\times64\) |
| regional head | \(b_R\) | \(M\) |
| RMR residual | \(r^{(t)}\) | \(1\times64\times64\) |
| final | \(Y^*\) | \(1\times64\times64\) |

Target:

\[
\boxed{
\text{Params}<0.30\text{M}
}
\]

preferred:

\[
\boxed{
\text{Params}<0.20\text{M}.
}
\]

This is a design target, not a paper result until measured.

---

# 27. Computational complexity

Let fine grid contain:

\[
G=H_gW_g
\]

cells and there be \(M\) rectangular regions.

### Forward regional operator

Build cumulative measure:

\[
O(G).
\]

Rectangle queries:

\[
O(M).
\]

### Adjoint

Difference-buffer corner updates:

\[
O(M).
\]

Prefix reconstruction:

\[
O(G).
\]

Therefore the exact regional residual machinery is approximately:

\[
\boxed{
O(G+M)
}
\]

per reconciliation iteration.

This is the efficiency rationale.

The paper must still report **measured latency**, because theoretical operation count does not guarantee hardware speed.

---

# 28. Relationship to PS-FH-CMICF

PS-FH is important project evidence, but it must **not become the field-level gap**.

## 28.1 What PS-FH demonstrates internally

The development path was:

\[
\text{global CMICF}
\rightarrow
\text{FH-CMICF}
\rightarrow
\text{PS-FH-CMICF}.
\]

On ShanghaiTech A:

### B5b global CMICF
- direct MAE: 209.63;
- RMSE: 308.52;
- direct/practical aggregate MAE gap: +91.32.

### B8 FH-CMICF, \(s=16,K=4\)
- direct MAE: 115.22;
- RMSE: 188.76;
- practical tiled MAE: 105.98;
- controlled tiled MAE: 103.23;
- direct/practical aggregate gap: +9.24.

### Current PS-FH-CMICF at epoch 500
- direct MAE: 104.64;
- best observed direct MAE by that point: 98.31;
- RMSE: 169.95;
- practical tiled MAE: 104.40;
- GAME(3): 172.49 versus 205.44 for B8;
- paired normalized direct/tiled discrepancy: approximately 4.06%.

Interpretation:

> **[I] Cumulative/count-structure manipulations carry useful signal, but making a learned cumulative field itself responsible for the primary fine prediction created repeated representational/conditioning problems.**

This is internal design evidence for moving from:

\[
F\to\hat C\to\Delta\hat C
\]

to:

\[
F\to Y
\quad\text{with}\quad
C=P(Y)
\]

as an exact internal state.

---

## 28.2 What PS-FH must not be used to claim

Do not write:

> “Existing crowd counters suffer from PS-FH phase bias.”

That is our model-specific pathology.

Do not write:

> “The field requires cumulative prediction because B8 improved.”

That is circular.

The field-level motivation must come from published literature.

PS-FH belongs in:

- development history;
- matched cumulative baseline;
- supplementary analysis;
- possibly one main ablation if page budget allows.

---

# 29. Novelty boundary

## Not novel individually

| Idea | Status |
|---|---|
| density integration | prior art |
| regional count supervision | prior art |
| local count map | prior art |
| arbitrary rectangle sum | classical |
| integral images / prefix sums | classical |
| inclusion-exclusion | classical |
| region hierarchy | prior art |
| iterative crowd refinement | prior art |
| uncertainty weighting | prior art |
| multi-scale fusion | prior art |
| scale routing | prior art |
| generic deep unrolling | prior art |
| generic forward/adjoint operators in neural networks | prior art |

---

## Provisional novel core

The strongest remaining claim is:

\[
\boxed{
\text{fine crowd measure}
\rightarrow
\text{explicit overlapping regional measurement space}
\rightarrow
\text{regional visual discrepancy}
\rightarrow
\text{exact adjoint back-projection}
\rightarrow
\text{fine measure correction}.
}
\]

Safe wording before the final collision search:

> **We did not find a prior crowd-counting method that performs this exact inference-time regional-measure reconciliation.**

Do not write “the first” until the final search is complete.

---

# 30. Mandatory matched experiment matrix

This is the central scientific experiment.

All variants must use:

- same encoder;
- same width;
- same data;
- same augmentation;
- same optimizer;
- same training schedule;
- same primary supervision;
- same output stride;
- matched or explicitly reported parameters/MACs.

## E0 — carrier

\[
B_0:
\quad
F\rightarrow Y.
\]

Direct stride-4 measure model.

---

## E1 — training-only regional information

\[
B_1:
\quad
B_0+L_R.
\]

No regional branch at inference.

Tests whether regional supervision alone solves the issue.

---

## E2 — regional evidence without reconciliation

\[
B_2:
\quad
B_0+H_R.
\]

Regional branch is trained, but final fine measure is not updated from it.

Tests whether auxiliary multi-task learning explains gains.

---

## E3 — generic learned refinement

\[
B_3:
\quad
Y^{(1)}
=
Y^{(0)}
+
R_\theta(F,Y^{(0)}).
\]

Parameter/MAC matched to RMR.

This is the key reviewer control.

---

## E4 — RMR with one iteration

\[
B_4:
\quad
T=1.
\]

---

## E5 — RMR with two iterations

\[
B_5:
\quad
T=2.
\]

Canonical proposed model if it wins.

---

# 31. Mechanism ablations

## M1 — remove exact adjoint

Replace:

\[
A^\top e
\]

with a learned projection of regional residuals.

If performance is unchanged, the exact geometry is not important.

---

## M2 — shuffled regional evidence

Randomly permute:

\[
b_R
\]

among regions at evaluation/training diagnostics.

If performance does not degrade, the regional evidence is not being used meaningfully.

---

## M3 — oracle regional evidence

Use ground-truth regional counts only as an **analysis upper bound**, never as normal inference.

This estimates whether the reconciliation mechanism has headroom independent of regional-head quality.

---

## M4 — region scales

Test:

\[
\{32\},
\quad
\{32,64\},
\quad
\{32,64,128\},
\quad
\{16,32,64,128\}.
\]

---

## M5 — overlap

Compare:

- non-overlapping partition;
- overlapping regular windows.

This tests whether overdetermined regional constraints are useful.

---

## M6 — iteration count

\[
T\in\{0,1,2,3\}.
\]

---

# 32. RQ-to-experiment map

| Research Question | Decisive comparisons |
|---|---|
| RQ1: exact operator vs learned context under ultra-light budget | \(B_3\) vs \(B_4/B_5\), plus Params/MACs/latency |
| RQ2: inference-time region information vs training-only | \(B_1,B_2\) vs \(B_4/B_5\) |
| mechanism: is \(A^\top\) meaningful? | exact \(A^\top\) vs learned projection; shuffled/oracle \(b_R\) |

The paper should be organized around these comparisons, not around a sequence of module additions.

---

# 33. Kill rules

A CVPR-quality project needs explicit failure conditions.

## K1 — central inference claim fails

If:

\[
\mathrm{MAE}(B_5)
\ge
\mathrm{MAE}(B_2)
\]

under matched training,

then inference-time reconciliation adds no value beyond the regional branch.

**Kill or reframe the central claim.**

---

## K2 — exact operator has no advantage

If same-cost CNN refinement:

\[
B_3
\]

matches or beats RMR:

\[
B_5,
\]

then the operator-driven inference claim is weak.

Do not hide this result.

---

## K3 — regional supervision is sufficient

If:

\[
B_1\approx B_5,
\]

then inference-time use of regional counts is unnecessary.

The paper should revert to a training-objective story or be abandoned.

---

## K4 — no accuracy benefit

If RMR improves only:

- GAME;
- regional consistency;
- direct/tiled stability;

but not standard MAE/RMSE,

then the method does not yet justify a top-tier counting paper.

---

## K5 — efficiency story fails

If the implementation has low theoretical MACs but poor measured latency because prefix/scatter operators are memory-bound, then “efficient inference” must be weakened.

Report actual hardware results.

---

# 34. Benchmark plan

## Primary datasets

1. ShanghaiTech Part A
2. UCF-QNRF
3. NWPU-Crowd

## Additional

4. ShanghaiTech Part B
5. JHU-CROWD++

The main claim is counting + efficiency, not localization.

Do not make a SOTA localization claim unless a separate localization protocol is implemented correctly.

---

# 35. Main metrics

## Counting

\[
\mathrm{MAE}
=
\frac1n\sum_i|\hat N_i-N_i|
\]

\[
\mathrm{RMSE}
=
\sqrt{
\frac1n\sum_i(\hat N_i-N_i)^2
}.
\]

NAE where appropriate.

---

## Efficiency

Report:

- parameters;
- MACs/FLOPs with explicit convention;
- batch-1 latency;
- p50/p95 latency if deployment is emphasized;
- peak memory;
- named hardware.

Do not compare latency measured under incompatible hardware as if it were directly comparable.

---

## Mechanism analysis

Use:

- GAME(0–3);
- regional MAE by region scale;
- residual magnitude before/after reconciliation;
- density-stratified MAE;
- background/empty-region error.

These are mechanism diagnostics, not replacements for MAE/RMSE.

---

# 36. Lightweight baseline set

At minimum include:

### Ultra-/lightweight
- TinyCount;
- LRMBNet;
- FusionCount;
- representative MobileNet/GhostNet lightweight counters where protocols are comparable.

### Strong supervision/reference
- DM-Count;
- PML-based counter;
- ZIP where relevant.

### Strong modern counting context
- representative recent high-accuracy methods, clearly separated from lightweight models.

The paper should contain an **accuracy–Params/MACs Pareto plot**, not only a leaderboard table.

---

# 37. Internal performance gates

These are project decisions, not scientific facts.

### Survival gate

\[
\boxed{
\mathrm{SHA\text{-}A\ MAE}<60
}
\]

with:

\[
\text{Params}<0.30M.
\]

### Strong result

\[
\boxed{
\mathrm{MAE}<55
}
\]

at the same budget.

### Aspirational

\[
\boxed{
48\text{–}52
}
\]

while remaining genuinely lightweight.

Failure to hit these targets does not automatically make the RQ false, but it changes the venue/claim strength.

---

# 38. CVPR-style paper identity

## Recommended working title

### Primary

**Regional Measure Reconciliation for Ultra-Light Crowd Counting**

### Alternative

**Learning Less Context: Operator-Guided Regional Reconciliation for Ultra-Light Crowd Counting**

### Avoid

- “Novel Cumulative Integral Network”
- “Unified Multi-Scale Cumulative Attention...”
- titles listing many modules.

The title should communicate one idea.

---

# 39. One-sentence paper thesis

> **An ultra-light crowd counter need not learn all regional count interactions from scratch: regional summation and its adjoint are known exactly, so learned visual capacity can be reserved for observing people and allocating mathematically defined regional corrections.**

This is the conceptual thesis.

It must be validated, not assumed.

---

# 40. Draft abstract — no fabricated results

> **Crowd counting presents an unusual efficiency challenge: most individuals occupy small image regions, making fine local evidence important, yet robust counting also benefits from regional and multi-scale context. Existing methods inject regional structure mainly through learned contextual modules, local-count supervision, or hierarchical prediction, all of which either consume model capacity or influence the model primarily during training. We ask whether part of this regional reasoning can instead be supplied by the algebra of counting itself. We formulate crowd estimation as regional measure reconciliation. A lightweight network first predicts a fine non-negative counting measure and independent visual counts for overlapping regions. The fine measure is projected into the same regional measurement space using an exact regional-sum operator; regional disagreements are then mapped back to the fine grid through the operator's exact adjoint, and a small learned preconditioner allocates the resulting correction using local visual evidence. Both the forward and adjoint regional operators admit cumulative-sum implementations, enabling multi-scale reconciliation with low overhead. We evaluate whether this inference-time formulation provides benefits beyond regional supervision and same-cost learned refinement under matched ultra-light budgets. [RESULTS TO BE INSERTED ONLY AFTER FINAL EXPERIMENTS.]**

This abstract is intentionally result-free until experiments exist.

---

# 41. Clean Introduction structure

A CVPR introduction should be approximately 5–6 paragraphs.

## Paragraph 1 — task and practical tension

Do not begin with the proposed method.

Establish:

- crowd counting is important;
- edge/real-time scenarios motivate compact models;
- dense crowds demand accuracy under large scale/density variation.

End with the tension:

> stronger contextual modeling usually costs compute, while aggressive compression hurts difficult dense scenes.

---

## Paragraph 2 — why local evidence matters

Use LIMM/ECAI and related evidence.

Point:

> heads are often very small, so a lightweight counter cannot afford to discard fine spatial evidence merely to enlarge receptive field.

Do not say “global context is unnecessary.”

---

## Paragraph 3 — regional counts are known to be useful, but prior use has a boundary

Survey in one paragraph:

- LCM;
- grid/local-count loss;
- Composition Loss;
- ChfL/GCFL;
- S-DCNet / regional counting.

Then state:

> prior work shows regional count structure is valuable, but it is typically expressed as supervision, hierarchical prediction, or aggregation.

This is the setup for the gap.

---

## Paragraph 4 — gap + RQs

State:

> We study a different question: can regional count structure become a cheap inference operator for a fine prediction, rather than another learned context module or training-only regularizer?

Then give RQ1/RQ2 in prose.

---

## Paragraph 5 — method insight

Introduce one equation:

\[
r
=
A^\top W(AY-b).
\]

Explain:

- \(AY\): counts implied by current fine measure;
- \(b\): independently predicted visual regional counts;
- \(A^\top\): exact back-projection.

Do not introduce every architectural detail in the introduction.

---

## Paragraph 6 — contributions

Three contributions maximum.

---

# 42. Contributions — clean version

Do not list backbone, fusion, uncertainty, and losses as separate contributions.

Use:

### Contribution 1 — formulation

> **We formulate lightweight crowd counting as regional measure reconciliation, in which a fine non-negative count measure is explicitly compared with independently inferred overlapping regional counts during inference.**

### Contribution 2 — operator-driven inference

> **We derive an efficient reconciliation update based on the exact regional-sum operator and its adjoint, separating known count geometry from learned visual allocation and allowing cumulative implementations with low overhead.**

### Contribution 3 — evidence

> **Through matched controls, we test whether inference-time reconciliation provides benefits beyond regional supervision and same-cost learned refinement, and evaluate the resulting accuracy–efficiency trade-off across standard crowd-counting benchmarks.**

Only after strong final results may Contribution 3 say “establishes a new Pareto frontier.”

---

# 43. Related Work — only three subsections

## 2.1 Efficient and local-first crowd counting

Discuss:

- TinyCount;
- FusionCount;
- LRMBNet;
- LIMM;
- representative lightweight works.

End:

> these works motivate the budget and local-first carrier but do not provide our reconciliation mechanism.

---

## 2.2 Regional and structured count supervision

Discuss:

- MESA;
- Composition Loss;
- LCM;
- local/grid losses;
- HMoDE relative local counting;
- ChfL/GCFL;
- S-DCNet / Divide and Count.

End:

> regional counts are established; the question is their inference-time role in correcting a fine measure.

---

## 2.3 Model-based / unrolled inference

Briefly cite:

- Learned Primal-Dual;
- operator-based learned reconstruction.

State explicitly:

> generic use of a forward operator and adjoint inside a learned iteration is not novel. Our contribution is the count-specific construction of a regional measurement space from the same image and its use for ultra-light crowd estimation.

This preempts reviewer criticism.

---

# 44. Method section structure

## 3.1 Setup

Define:

\[
I,\mathcal P,Y,A,\mathcal R.
\]

## 3.2 Fine and regional visual observations

Define:

\[
Y^{(0)},b_R.
\]

## 3.3 Regional measure reconciliation

Derive:

\[
E(Y)
=
\frac12\|W^{1/2}(AY-b)\|^2
\]

and:

\[
\nabla_YE
=
A^\top W(AY-b).
\]

Then positive update.

## 3.4 Efficient realization

Explain:

- prefix for \(A\);
- difference-buffer + prefix for \(A^\top\);
- complexity.

## 3.5 Training and lightweight instantiation

Encoder widths, losses, T.

No additional method subsection unless experimentally necessary.

---

# 45. Experiment section structure by RQ

## 4.1 Setup

Datasets, metrics, training, hardware, baselines.

## 4.2 RQ1 — does operator-driven context improve the ultra-light Pareto frontier?

Main comparison:

\[
B_0,\ B_3,\ B_5.
\]

Show:

- MAE/RMSE;
- params;
- MACs;
- latency.

## 4.3 RQ2 — is inference-time reconciliation different from regional supervision?

Compare:

\[
B_0,\ B_1,\ B_2,\ B_4,\ B_5.
\]

## 4.4 Why does it work?

Mechanism:

- residual before/after;
- GAME;
- regional scale error;
- exact \(A^\top\) vs learned projection;
- oracle/shuffled evidence.

## 4.5 Comparison with SOTA/lightweight methods

Pareto plot + table.

## 4.6 Limitations

Include failure cases and hardware caveats.

---

# 46. Main-paper figure plan

A CVPR-style main paper has limited space.

Based on the 8-page main-paper rule used by CVPR 2026, central evidence cannot be hidden in supplementary material.

## Figure 1 — motivation, not architecture clutter

Three panels:

1. fine count map with correct/incorrect local regions;
2. regional measurement residuals at multiple scales;
3. exact back-projection produces a correction field.

One visual equation:

\[
Y
\rightarrow
AY-b
\rightarrow
A^\top(AY-b)
\rightarrow
Y'.
\]

---

## Figure 2 — architecture

Only:

```text
local encoder
→ fine measure
→ regional measurement
↔ regional visual evidence
→ exact adjoint
→ corrected measure
```

No 20-box diagram.

---

## Figure 3 — mechanism

Show:

- \(Y^{(0)}\);
- regional error;
- adjoint residual;
- \(Y^{(1)}\);
- \(Y^*\).

This makes the contribution visually understandable.

---

# 47. Main tables

## Table 1 — lightweight/SOTA Pareto

Columns:

| Method | Params | MACs | SHA MAE/RMSE | QNRF MAE/RMSE | NWPU MAE/RMSE |

Separate ultra-light and heavy methods if needed.

---

## Table 2 — decisive RQ ablation

| Variant | Regional training | Regional inference | exact \(A^\top\) | Params | MAE | RMSE |

This is more important than a long component checklist.

---

## Table 3 — mechanism

| Method | regional MAE | GAME(1) | GAME(2) | GAME(3) | latency overhead |

---

# 48. Page-budget plan

Use an 8-page CVPR-style main-paper target.

Approximate budget:

| Section | Pages |
|---|---:|
| Abstract + Intro | 1.3 |
| Related Work | 0.7 |
| Method | 2.2 |
| Experiments | 3.2 |
| Limitations / conclusion | 0.6 |

References excluded from the 8-page content limit under CVPR 2026 rules.

The actual target conference guidelines must be re-checked at submission time.

---

# 49. Reviewer red-team

## Attack A

> Regional count supervision already exists.

Answer:

Correct. We do not claim regional count supervision. The decisive comparison is training-only regional supervision versus inference-time reconciliation.

---

## Attack B

> This is just another iterative refinement network.

Answer:

Generic refinement learns an unconstrained residual. RMR explicitly constructs the correction direction from:

\[
A^\top W(AY-b).
\]

The same-cost CNN refinement control determines whether this structure matters empirically.

---

## Attack C

> This is just deep unrolling.

Answer:

Generic operator-based unrolling is prior art and is cited. The proposed research contribution is the task-specific regional count measurement formulation, in which the regional “measurements” are independent visual estimates from the same image and the count operator defines the correction geometry under an ultra-light budget.

If reviewers do not accept that task-specific formulation as sufficiently distinct, the paper must rely on unusually strong empirical/efficiency evidence.

---

## Attack D

> Integral images are old.

Answer:

Agreed. Prefix sums are an implementation device, not the novelty claim.

---

## Attack E

> Why not use regional loss only?

Answer:

That is RQ2 and a mandatory baseline.

If regional loss matches RMR, the proposed inference mechanism fails its main hypothesis.

---

## Attack F

> Why not add another convolution?

Answer:

That is RQ1's matched same-cost refinement baseline.

---

## Attack G

> Your regional count head is simply a second head.

Answer:

The regional head by itself is explicitly ablated. The claim requires the transition:

\[
B_2\rightarrow B_5
\]

to provide significant additional benefit.

---

## Attack H

> Both \(Y\) and \(b_R\) come from the same image, so the “measurement” is not independent.

Answer:

“Independent” here must mean **independent readout**, not statistically independent random variables.

Use precise wording:

> “separately inferred regional visual evidence.”

Do not claim statistical independence.

---

# 50. Claims allowed before experiments

## Allowed now

- regional counting is established prior art;
- local information is important in crowd counting;
- lightweight/context efficiency is a documented challenge;
- regional sum is a linear operator;
- its adjoint is analytically defined;
- rectangle sums and adjoints admit cumulative implementations;
- we did not find an exact crowd-counting collision in the reviewed literature.

---

## Not allowed now

- “first”;
- “novel” without qualification;
- “SOTA”;
- “more efficient than attention” without measured hardware results;
- “better than regional loss” before matched experiment;
- “exact operators improve generalization” before evidence;
- “cumulative reasoning solves dense counting.”

---

# 51. Conditional paper claims

Only if experiments support them.

## C1 — mechanism

If:

\[
B_5<B_2
\]

and:

\[
B_5<B_3
\]

with meaningful margin:

> inference-time regional reconciliation contributes beyond regional supervision and generic refinement.

---

## C2 — efficiency

If RMR improves accuracy at matched/smaller Params, MACs, and measured latency:

> explicit count operators improve the accuracy–efficiency trade-off.

---

## C3 — Pareto/SOTA

Only if verified against current methods under fair protocols:

> RMR-Count establishes a new ultra-light accuracy–efficiency Pareto frontier.

---

# 52. What would make this a CVPR paper rather than engineering?

The paper needs all four:

1. **a literature-grounded question**, not only a new module;
2. **a precise mathematical formulation**:
   \[
   A,\ A^\top,\ E(Y);
   \]
3. **a decisive control that could falsify the contribution**:
   same-cost learned refinement;
4. **strong empirical consequence**:
   accuracy/efficiency improvement on multiple datasets.

A new backbone plus several modules is not enough.

---

# 53. What would kill CVPR-level ambition?

Any of:

- novelty collision with an existing count-reconciliation method;
- CRR/RMR only improves internal diagnostics;
- same-cost CNN gives the same gain;
- regional training-only loss gives the same gain;
- large absolute gap to lightweight SOTA;
- method requires >1–2M parameters to work;
- theoretical “efficiency” does not translate to measured latency;
- paper requires five loosely connected modules to become competitive.

If that happens, reframe or stop.

---

# 54. Immediate implementation sequence

Do not implement the old large UL-CMR bundle.

## Phase 0 — exact operators

Implement and test:

```text
regional_sum_A
regional_adjoint_AT
prefix2d
difference_buffer_backprojection
```

Unit test adjoint identity:

\[
\boxed{
\langle AY,e\rangle
=
\langle Y,A^\top e\rangle.
}
\]

---

## Phase 1 — carrier ceiling

Build:

\[
B_0.
\]

The carrier must already be competitive enough that the RMR effect is meaningful.

A weak carrier cannot validate a SOTA architecture.

---

## Phase 2 — RQ2 controls

Run:

\[
B_1,\ B_2,\ B_4,\ B_5.
\]

---

## Phase 3 — RQ1 control

Run same-cost:

\[
B_3.
\]

This is mandatory before adding more modules.

---

## Phase 4 — only then improve carrier

If RMR survives:

- strengthen local feature fusion;
- improve regional evidence;
- test strong standard supervision;
- optimize kernels/latency.

Do not change the central mechanism.

---

# 55. Suggested repository structure

```text
hpc/
  models/
    rmr_encoder.py
    rmr_count.py
  operators/
    regional_measure.py
  losses/
    rmr_losses.py

tests/
  test_regional_measure_operator.py
  test_regional_adjoint.py
  test_rmr_update.py

configs/
  rmr/
    b0_direct.yaml
    b1_region_loss.yaml
    b2_region_head.yaml
    b3_cnn_refine.yaml
    b4_rmr_t1.yaml
    b5_rmr_t2.yaml

tools/
  train_rmr.py
  eval_rmr.py
  profile_rmr.py
```

---

# 56. Mandatory mathematical unit tests

## Forward rectangle exactness

For random \(Y\):

\[
(AY)_m
=
\sum_{p\in R_m}Y_p.
\]

Compare operator output to explicit summation.

---

## Adjoint identity

For random \(Y,e\):

\[
\left|
\langle AY,e\rangle
-
\langle Y,A^\top e\rangle
\right|
<10^{-10}
\]

in float64.

---

## Positivity

For every iteration:

\[
Y^{(t)}\ge0.
\]

---

## Global conservation

\[
\sum Y
=
P(Y)[-1,-1].
\]

---

## Zero residual fixed point

If:

\[
AY=b,
\]

then:

\[
r=0.
\]

The reconciliation update must not change \(Y\) except numerical noise.

---

# 57. Literature table for the final Related Work

| Work | What it establishes | Why it does not close the proposed gap |
|---|---|---|
| Lempitsky & Zisserman 2010 | density/region count formulation; MESA | training objective, not inference regional residual reconciliation |
| Composition Loss, ECCV 2018 | multi-task count/density/localization consistency | no exact regional residual adjoint update |
| Iterative Crowd Counting, ECCV 2018 | learned iterative density refinement | correction learned generically |
| S-DCNet, ICCV 2019 | inference-time hierarchical local counting | divide/merge local predictions, not fine-measure residual backprojection |
| LCM, ECCV 2020 | local count map and train/eval alignment | regional information mainly as target/loss |
| HMoDE, TIP 2023 | multi-scale experts + relative local count | regional relation used for training; fusion learned |
| ChfL, CVPR 2022 / GCFL TPAMI | all-subregion-aware supervision | training loss, not inference reconciliation |
| FusionCount, ICIP 2022 | efficient use of multi-scale encoder features | learned feature fusion |
| TinyCount, 2024 | ultra-light edge counter | no operator-driven regional inference |
| LIMM, ECAI 2025 | local-first principle for crowd counting | different model-design principle |
| PML, ICLR 2025 | principled loss derived from representation assumption | training objective, different problem |
| Learned Primal-Dual, TMI 2018 | generic forward/adjoint learned unrolling | inverse reconstruction; not crowd regional count formulation |
| Box Conv / SAT Conv | efficient integral-image neural context | integral operator use itself already known |

---

# 58. Reference links for the working draft

1. Pan & Jia, **Local Information Matters: A Rethink of Crowd Counting**, ECAI 2025.  
   DOI: https://doi.org/10.3233/FAIA250799

2. Liu, Yang, Ding, **Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting**, ECCV 2020.  
   https://arxiv.org/abs/2005.05776

3. Du et al., **Redesigning Multi-Scale Neural Network for Crowd Counting**, IEEE TIP 2023.  
   https://arxiv.org/abs/2208.02894

4. Shu et al., **Crowd Counting in the Frequency Domain**, CVPR 2022.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Shu_Crowd_Counting_in_the_Frequency_Domain_CVPR_2022_paper.html

5. Shu, Wan, Chan, **Generalized Characteristic Function Loss for Crowd Analysis in the Frequency Domain**, TPAMI 2024.  
   DOI: https://doi.org/10.1109/TPAMI.2023.3336196

6. Xiong et al., **From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer**, ICCV 2019.  
   https://openaccess.thecvf.com/content_ICCV_2019/html/Xiong_From_Open_Set_to_Closed_Set_Counting_Objects_by_Spatial_ICCV_2019_paper.html

7. Ranjan, Le, Hoai, **Iterative Crowd Counting**, ECCV 2018.  
   https://openaccess.thecvf.com/content_ECCV_2018/html/Viresh_Ranjan_Iterative_Crowd_Counting_ECCV_2018_paper.html

8. Idrees et al., **Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds**, ECCV 2018.  
   https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Haroon_Idrees_Composition_Loss_for_ECCV_2018_paper.php

9. Ma, Sánchez, Guha, **FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion**, ICIP 2022.

10. Lee & Lee, **TinyCount: An Efficient Crowd Counting Network for Intelligent Surveillance**, JRTIP 2024.  
    DOI: https://doi.org/10.1007/s11554-024-01531-8

11. Li, Zheng, Lu, **Lightweight Res-Connection Multi-Branch Network for Highly Accurate Crowd Counting and Localization**, CMC 2024.  
    https://www.techscience.com/cmc/v79n2/56427

12. Lin, Wan, Chan, **Proximal Mapping Loss: Understanding Loss Functions in Crowd Counting & Localization**, ICLR 2025.  
    https://proceedings.iclr.cc/paper_files/paper/2025/file/04c956c52bfc1cc5f6dd989c729213b7-Paper-Conference.pdf

13. Adler & Öktem, **Learned Primal-Dual Reconstruction**, 2018.  
    https://arxiv.org/abs/1707.06474

14. Burkov & Lempitsky, **Deep Neural Networks with Box Convolutions**, NeurIPS 2018.

15. Zhang, Halber, Rusinkiewicz, **Accelerating Large-Kernel Convolution Using Summed-Area Tables**, 2019.  
    https://arxiv.org/abs/1906.11367

---

# 59. CVPR writing discipline learned from strong recent papers

The paper should follow the pattern used by strong formulation-driven work such as PML and recent CVPR papers:

```text
real limitation
    ↓
specific observation
    ↓
precise research question
    ↓
mathematical formulation
    ↓
minimal method
    ↓
controlled experiment
    ↓
broader SOTA evidence
```

Avoid:

```text
backbone
+ attention
+ module A
+ module B
+ special loss
+ another head
→ call the combination a contribution
```

---

# 60. Final paper frame

The clean logical chain is:

\[
\boxed{
\begin{aligned}
&\text{Crowd heads are predominantly local/small}\\
&+\ \text{regional count structure is useful}\\
&+\ \text{learned context is expensive under ultra-light budgets}\\
&\Downarrow\\
&\textbf{RQ1: can known count operators replace some learned context?}\\
&\textbf{RQ2: is regional information more useful at inference than only as supervision?}\\
&\Downarrow\\
&\text{fine non-negative measure }Y\\
&\xrightarrow{A}
\text{regional implied counts}\\
&\xrightarrow{-b}
\text{regional residual}\\
&\xrightarrow{A^\top}
\text{fine correction geometry}\\
&\xrightarrow{\text{small learned preconditioner}}
Y'\\
&\Downarrow\\
&\text{matched tests against regional-loss-only and same-cost CNN refinement.}
\end{aligned}
}
\]

This is the research paper.

Not “cumulative mathematics is novel.”

Not “regional counts are novel.”

Not “iterative refinement is novel.”

The contribution survives only if the **count-specific inference formulation** produces a measurable advantage under the ultra-light regime.

---

# 61. Final decision rule

Proceed to full RMR-Count only if the first matched experiment establishes:

\[
\boxed{
B_5 < B_2
}
\]

and:

\[
\boxed{
B_5 < B_3
}
\]

with a practically meaningful MAE/RMSE gain at comparable efficiency.

If those inequalities do not hold, do not add more modules to rescue the architecture.

The Research Questions would have been answered negatively, and the correct scientific action is to reformulate the thesis.

---

# 62. Bottom line

The clean CVPR thesis is not:

> “We found a new cumulative representation.”

It is:

> **“Under a severe capacity budget, crowd counting should not require a neural network to relearn regional count algebra. We test whether explicit regional measurement and exact residual back-projection can turn known count structure into useful inference while preserving fine local evidence.”**

That thesis comes from the intersection of documented limitations in the field, has a clear prior-art boundary, and can be falsified with matched experiments.
