

> **Venue standard:** CVPR is treated here as an A-tier/top-tier computer-vision conference. The design below is therefore not framed as “good enough to publish”; it is built around a literature-grounded limitation, falsifiable RQs, a single mathematical mechanism, matched controls, multi-dataset evidence, efficiency measurement, and explicit kill rules.

> **No fabricated results:** all numerical results in the historical PS-FH/R2 sections are prior project evidence. All RMR-Count results are placeholders until the corresponding experiment is run. No “SOTA”, “first”, or causal claim is permitted before the registered controls pass.

---

# A. Updated 2026 collision check and paper boundary

The current literature check includes recent 2025–2026 work such as ZIP (zero-inflated Poisson block counts), Local Information Matters, CVPR 2026 lightweight image-to-video counting, recent regional calibration methods, PML, ChfL/GCFL, S-DCNet, local-count objectives, iterative crowd refinement, and generic learned forward/adjoint unrolling.

The paper **must not** claim novelty from any of the following:

- regional count supervision;
- local counting maps;
- arbitrary rectangle counting;
- summed-area tables / integral images;
- inclusion–exclusion;
- iterative refinement;
- multi-scale feature fusion;
- local-first modeling;
- generic model-based unrolling;
- using a known forward operator and its adjoint inside a neural network.

The provisional surviving contribution is narrower:

\[
\boxed{
\text{fine non-negative crowd measure}
\to
\text{overlapping regional count measurement space}
\to
\text{visual regional disagreement}
\to
\text{exact count-operator adjoint}
\to
\text{fine-grid correction at inference}
}
\]

The exact collision claim remains:

> **As of the updated search used for this specification, we did not find a crowd-counting method with this exact inference-time regional-measure reconciliation formulation. This is a search result, not permission to write “the first” in the final paper. A final collision search is mandatory immediately before submission.**

---

# RMR-Count: Regional Measure Reconciliation for Ultra-Light Crowd Counting
## Complete CVPR A-tier paper specification + equations + executable PyTorch reference implementation + evaluation protocol

**Working method name:** RMR-Count  
**Status:** complete research-and-implementation specification — claims remain conditional on the registered experiments  
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

### [I] Inference / synpaper claim
A conclusion drawn by comparing multiple papers. It is plausible but is not a quotation or explicit claim from one paper.

### [P] Proposed hypopaper claim or method
A new hypopaper claim, design, or planned experiment from this project.

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

### Falsifiable hypopaper claim H1

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

### Falsifiable hypopaper claim H2

Let:

- \(B\): direct fine-measure model;
- \(B+L_R\): same model with regional-count supervision only;
- \(B+H_R\): regional count head but no reconciliation;
- \(B+\mathrm{RMR}\): regional head plus inference-time reconciliation.

The hypopaper claim is:

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

# 39. One-sentence central paper claim

> **An ultra-light crowd counter need not learn all regional count interactions from scratch: regional summation and its adjoint are known exactly, so learned visual capacity can be reserved for observing people and allocating mathematically defined regional corrections.**

This is the conceptual paper claim.

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

If regional loss matches RMR, the proposed inference mechanism fails its main hypopaper claim.

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

The Research Questions would have been answered negatively, and the correct scientific action is to reformulate the paper.

---

# 62. Bottom line

The clean CVPR paper claim is not:

> “We found a new cumulative representation.”

It is:

> **“Under a severe capacity budget, crowd counting should not require a neural network to relearn regional count algebra. We test whether explicit regional measurement and exact residual back-projection can turn known count structure into useful inference while preserving fine local evidence.”**

That paper claim comes from the intersection of documented limitations in the field, has a clear prior-art boundary, and can be falsified with matched experiments.



# 63. Executable reference implementation

This section contains the complete reference implementation used by the registered experiment matrix. The code has been syntax-checked and the operator/model unit tests below pass.

## 63.1 Environment

Recommended reproducible environment:

```bash
python >= 3.10
PyTorch >= 2.2
CUDA 12.x
1× NVIDIA T4 16 GB for the project default hardware study
```

Install:

```bash
pip install -r requirements.txt
```

The implementation intentionally uses **one GPU only**. Do not use DDP for the canonical latency or causal ablation runs.

## 63.2 File tree

```text
rmr_count_reference/
├── requirements.txt
├── run_matrix.sh
├── configs/
│   ├── direct.yaml
│   ├── region_loss.yaml
│   ├── region_aux.yaml
│   ├── learned_project.yaml
│   ├── rmr_t1.yaml
│   └── rmr_t2.yaml
├── rmr_count/
│   ├── __init__.py
│   ├── operators.py
│   ├── model.py
│   ├── losses.py
│   ├── data.py
│   ├── prepare_manifest.py
│   ├── split_manifest.py
│   ├── metrics.py
│   ├── train.py
│   ├── eval.py
│   ├── profile.py
│   └── aggregate.py
└── tests/
    ├── test_operators.py
    ├── test_model.py
    └── test_data.py
```

## 63.3 Registered model variants

| ID | `variant` | Regional training signal | Regional head at inference | Projection/refinement |
|---|---|---|---|---|
| B0 | `direct` | no | no | none |
| B1 | `region_loss` | map regional loss | no | none |
| B2 | `region_aux` | regional head loss | yes, auxiliary only | none |
| B3 | `learned_project` | same regional head loss | yes | learned center-scatter projector |
| B4 | `rmr`, T=1 | same regional head loss | yes | exact \(A^\top\), one step |
| B5 | `rmr`, T=2 | same regional head loss | yes | exact \(A^\top\), two steps |

The strongest causal comparison is **B3 vs B5** because both consume separately inferred regional evidence, but B3 must learn the region-to-grid correction geometry whereas B5 receives the exact count-operator adjoint.

Current reference parameter counts with width 32 are approximately:

```text
direct           58,867
region_loss      58,867
region_aux       63,140
learned_project  70,597
rmr              64,677
```

Thus the learned-project control is slightly **larger** than RMR rather than artificially handicapped. Final paper tables must report measured values from the exact committed code.

---

## File: `requirements.txt`

```text
torch>=2.2
torchvision>=0.17
numpy>=1.26
Pillow>=10.0
PyYAML>=6.0
scipy>=1.11
pytest>=8.0
```


## File: `rmr_count/operators.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionSet:
    """Rectangular regions on a feature/count grid.

    boxes: [M, 4] int64 with half-open coordinates (y1, x1, y2, x2).
    scale_id: [M] int64 index of the image-pixel region scale; -1 for full image.
    area: [M] float count-grid area.
    """

    boxes: torch.Tensor
    scale_id: torch.Tensor
    area: torch.Tensor

    def to(self, device: torch.device | str) -> "RegionSet":
        return RegionSet(
            boxes=self.boxes.to(device),
            scale_id=self.scale_id.to(device),
            area=self.area.to(device),
        )


def prefix2d(x: torch.Tensor) -> torch.Tensor:
    """Inclusive 2-D prefix sum with a zero top row/left column.

    Input:  [B, C, H, W]
    Output: [B, C, H+1, W+1]
    """
    if x.ndim != 4:
        raise ValueError(f"prefix2d expects [B,C,H,W], got {tuple(x.shape)}")
    p = x.cumsum(dim=-2).cumsum(dim=-1)
    return F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)


def _gather_prefix(prefix: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gather prefix values at M coordinates for every batch/channel."""
    b, c, hp, wp = prefix.shape
    idx = (y * wp + x).view(1, 1, -1).expand(b, c, -1)
    return torch.gather(prefix.flatten(-2), dim=-1, index=idx)


def rectangle_sum_from_prefix(prefix: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Rectangle sums using a padded prefix table.

    prefix: [B,C,H+1,W+1]
    boxes:  [M,4] in half-open grid coordinates
    returns [B,C,M]
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [M,4]")
    boxes = boxes.long()
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    br = _gather_prefix(prefix, y2, x2)
    tr = _gather_prefix(prefix, y1, x2)
    bl = _gather_prefix(prefix, y2, x1)
    tl = _gather_prefix(prefix, y1, x1)
    return br - tr - bl + tl


def regional_sum(x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Linear regional-count operator A: [B,C,H,W] -> [B,C,M]."""
    return rectangle_sum_from_prefix(prefix2d(x), boxes)


def regional_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Exact adjoint A^T of rectangular summation.

    values: [B,C,M]
    boxes:  [M,4]
    returns [B,C,H,W]

    Uses a 2-D difference buffer followed by cumulative sums.
    """
    if values.ndim != 3:
        raise ValueError(f"values must be [B,C,M], got {tuple(values.shape)}")
    b, c, m = values.shape
    if boxes.shape != (m, 4):
        raise ValueError(f"boxes must be [{m},4], got {tuple(boxes.shape)}")

    boxes = boxes.long()
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    hp, wp = height + 1, width + 1

    diff = values.new_zeros((b, c, hp * wp))

    def scatter(y: torch.Tensor, x: torch.Tensor, src: torch.Tensor) -> None:
        idx = (y * wp + x).view(1, 1, -1).expand(b, c, -1)
        diff.scatter_add_(dim=-1, index=idx, src=src)

    scatter(y1, x1, values)
    scatter(y1, x2, -values)
    scatter(y2, x1, -values)
    scatter(y2, x2, values)

    diff = diff.view(b, c, hp, wp)
    field = diff.cumsum(dim=-2).cumsum(dim=-1)
    return field[..., :height, :width]


def _axis_starts(length: int, window: int, step: int) -> list[int]:
    if window >= length:
        return [0]
    starts = list(range(0, max(1, length - window + 1), max(1, step)))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def build_multiscale_regions(
    height: int,
    width: int,
    output_stride: int,
    region_sizes_px: Sequence[int] = (16, 32, 64, 128),
    overlap: float = 0.5,
    include_full_image: bool = True,
    device: torch.device | str | None = None,
) -> RegionSet:
    """Build deterministic overlapping rectangular regions.

    Region sizes are specified in image pixels and quantized to the output grid.
    The last window on each axis is forced to touch the image/grid boundary.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    boxes: list[tuple[int, int, int, int]] = []
    scale_ids: list[int] = []

    for sid, size_px in enumerate(region_sizes_px):
        win = max(1, int(round(size_px / output_stride)))
        wy = min(win, height)
        wx = min(win, width)
        sy = max(1, int(round(wy * (1.0 - overlap))))
        sx = max(1, int(round(wx * (1.0 - overlap))))
        ys = _axis_starts(height, wy, sy)
        xs = _axis_starts(width, wx, sx)
        for y1 in ys:
            for x1 in xs:
                boxes.append((y1, x1, y1 + wy, x1 + wx))
                scale_ids.append(sid)

    if include_full_image:
        full = (0, 0, height, width)
        if full not in boxes:
            boxes.append(full)
            scale_ids.append(-1)

    box_t = torch.tensor(boxes, dtype=torch.long, device=device)
    scale_t = torch.tensor(scale_ids, dtype=torch.long, device=device)
    area_t = ((box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])).float()
    return RegionSet(boxes=box_t, scale_id=scale_t, area=area_t)


def region_geometry(
    boxes: torch.Tensor,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Geometry features [M,6]: cy,cx,h,w,log_area,log_aspect."""
    boxes = boxes.float()
    y1, x1, y2, x2 = boxes.unbind(-1)
    h = (y2 - y1).clamp_min(1.0)
    w = (x2 - x1).clamp_min(1.0)
    cy = 0.5 * (y1 + y2) / max(float(height), 1.0)
    cx = 0.5 * (x1 + x2) / max(float(width), 1.0)
    hn = h / max(float(height), 1.0)
    wn = w / max(float(width), 1.0)
    area = (h * w) / max(float(height * width), 1.0)
    aspect = w / (h + eps)
    return torch.stack([cy, cx, hn, wn, torch.log(area + eps), torch.log(aspect + eps)], dim=-1)


def region_average_features(features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Average pooled region features: [B,C,H,W] -> [B,M,C]."""
    sums = regional_sum(features, boxes)  # [B,C,M]
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).to(features.dtype)
    avg = sums / area.view(1, 1, -1).clamp_min(1.0)
    return avg.transpose(1, 2).contiguous()


def center_scatter(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sparse learned-projection control: place each region residual at its center.

    values: [B,1,M]
    returns [B,1,H,W] with collision averaging.
    """
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError("center_scatter expects values [B,1,M]")
    b, _, m = values.shape
    y = ((boxes[:, 0] + boxes[:, 2] - 1) // 2).long().clamp(0, height - 1)
    x = ((boxes[:, 1] + boxes[:, 3] - 1) // 2).long().clamp(0, width - 1)
    idx = (y * width + x).view(1, 1, m).expand(b, 1, -1)
    out = values.new_zeros((b, 1, height * width))
    cnt = values.new_zeros((b, 1, height * width))
    out.scatter_add_(-1, idx, values)
    cnt.scatter_add_(-1, idx, torch.ones_like(values))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, 1, height, width)
```


## File: `rmr_count/model.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)

Variant = Literal[
    "direct",
    "region_loss",
    "region_aux",
    "learned_project",
    "rmr",
]


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1, act: bool = True):
        pad = k // 2
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=False), _gn(cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class TinyIR(nn.Module):
    """Small inverted residual block using depthwise spatial mixing."""

    def __init__(self, cin: int, cout: int, stride: int = 1, expand: float = 2.0):
        super().__init__()
        mid = max(cin, int(round(cin * expand)))
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvGNAct(cin, mid, k=1) if mid != cin else nn.Identity()
        self.dw = ConvGNAct(mid, mid, k=3, stride=stride, groups=mid)
        self.proj = nn.Sequential(nn.Conv2d(mid, cout, 1, bias=False), _gn(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.dw(self.expand(x)))
        return x + y if self.use_res else y


class TinyLocalEncoder(nn.Module):
    """Local-first encoder exposing stride-4/8/16 features."""

    def __init__(self):
        super().__init__()
        self.stem = ConvGNAct(3, 16, 3, stride=2)
        self.s4 = nn.Sequential(TinyIR(16, 24, stride=2), TinyIR(24, 24))
        self.s8 = nn.Sequential(TinyIR(24, 40, stride=2), TinyIR(40, 40), TinyIR(40, 40))
        self.s16 = nn.Sequential(TinyIR(40, 64, stride=2), TinyIR(64, 64))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c4 = self.s4(x)
        c8 = self.s8(c4)
        c16 = self.s16(c8)
        return c4, c8, c16


class AdditiveFusion(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.p4 = ConvGNAct(24, width, 1)
        self.p8 = ConvGNAct(40, width, 1)
        self.p16 = ConvGNAct(64, width, 1)
        self.out = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
        )

    def forward(self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor) -> torch.Tensor:
        size = c4.shape[-2:]
        p = self.p4(c4)
        p = p + F.interpolate(self.p8(c8), size=size, mode="bilinear", align_corners=False)
        p = p + F.interpolate(self.p16(c16), size=size, mode="bilinear", align_corners=False)
        return self.out(p)


class FineMeasureHead(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.body(f)


class RegionalEvidenceHead(nn.Module):
    """Shared region-count regressor over integral-feature pooled descriptors."""

    def __init__(self, feature_dim: int = 32, hidden: int = 48, geom_dim: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + geom_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, f: torch.Tensor, regions: RegionSet) -> torch.Tensor:
        b, c, h, w = f.shape
        pooled = region_average_features(f, regions.boxes)  # [B,M,C]
        geom = region_geometry(regions.boxes, h, w).to(dtype=f.dtype)
        geom = geom.unsqueeze(0).expand(b, -1, -1)
        raw = self.mlp(torch.cat([pooled, geom], dim=-1)).squeeze(-1)
        return F.softplus(raw).unsqueeze(1)  # [B,1,M]


class LocalPreconditioner(nn.Module):
    def __init__(self, feature_dim: int = 32, hidden: int = 32, m_min: float = 0.25, m_max: float = 1.75):
        super().__init__()
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.net = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f: torch.Tensor, y: torch.Tensor, residual_field: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(torch.cat([f, y, residual_field], dim=1)))
        return self.m_min + (self.m_max - self.m_min) * gate


class LearnedRegionProjector(nn.Module):
    """Control that must learn how sparse region-center residuals affect the fine grid."""

    def __init__(self, feature_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 3),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            ConvGNAct(hidden, hidden, 3),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f: torch.Tensor, y: torch.Tensor, sparse_residual: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f, y, sparse_residual], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    region_sizes_px: tuple[int, ...] = (16, 32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = True
    iterations: int = 2
    eta_max: float = 1.0
    eps: float = 1e-6


class RMRCount(nn.Module):
    """Regional Measure Reconciliation crowd counter.

    Core variant `rmr`:
        image -> fine non-negative measure Y0
              -> separately inferred region counts b
              -> q = A Y
              -> exact residual back-projection A^T[(q-b)/area]
              -> small learned local preconditioner
              -> positive latent update

    Baseline variants are implemented in the same class for matched experiments.
    """

    def __init__(self, cfg: RMRConfig = RMRConfig(), variant: Variant = "rmr"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.encoder = TinyLocalEncoder()
        self.fusion = AdditiveFusion(cfg.feature_width)
        self.fine_head = FineMeasureHead(cfg.feature_width)

        needs_region_head = variant in {"region_aux", "learned_project", "rmr"}
        self.region_head = RegionalEvidenceHead(cfg.feature_width) if needs_region_head else None
        self.preconditioner = LocalPreconditioner(cfg.feature_width) if variant == "rmr" else None
        self.learned_projector = LearnedRegionProjector(cfg.feature_width, hidden=16) if variant == "learned_project" else None

        n_steps = max(1, cfg.iterations)
        self.eta_logits = nn.Parameter(torch.zeros(n_steps))

    def _regions(self, h: int, w: int, device: torch.device) -> RegionSet:
        return build_multiscale_regions(
            height=h,
            width=w,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=self.cfg.include_full_image,
            device=device,
        )

    def _eta(self, t: int) -> torch.Tensor:
        idx = min(t, self.eta_logits.numel() - 1)
        return self.cfg.eta_max * torch.sigmoid(self.eta_logits[idx])

    def _normalized_region_error(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        q = regional_sum(y, regions.boxes)  # [B,1,M]
        area = regions.area.to(y.dtype).view(1, 1, -1)
        return (q - b_region) / area.clamp_min(1.0)

    def _rmr_field(self, y: torch.Tensor, b_region: torch.Tensor, regions: RegionSet) -> torch.Tensor:
        bsz, _, h, w = y.shape
        e = self._normalized_region_error(y, b_region, regions)
        back = regional_adjoint(e, regions.boxes, h, w)
        ones = torch.ones_like(e)
        coverage = regional_adjoint(ones, regions.boxes, h, w)
        return back / coverage.clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | RegionSet | list[torch.Tensor]]:
        c4, c8, c16 = self.encoder(x)
        f = self.fusion(c4, c8, c16)
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]
        regions = self._regions(h, w, x.device)

        out: dict[str, torch.Tensor | RegionSet | list[torch.Tensor]] = {
            "features": f,
            "z0": z0,
            "y0": y0,
            "regions": regions,
        }

        if self.region_head is not None:
            b_region = self.region_head(f, regions)
            out["b_region"] = b_region
        else:
            b_region = None

        if self.variant in {"direct", "region_loss", "region_aux"}:
            out["y"] = y0
            out["iterates"] = [y0]
            return out

        z = z0
        y = y0
        iterates = [y0]
        residual_fields: list[torch.Tensor] = []

        if b_region is None:
            raise RuntimeError(f"variant {self.variant} requires regional evidence")

        for t in range(self.cfg.iterations):
            if self.variant == "rmr":
                r = self._rmr_field(y, b_region, regions)
                residual_fields.append(r)
                assert self.preconditioner is not None
                m = self.preconditioner(f, y, r)
                # Chain rule for Y=softplus(z): dY/dz = sigmoid(z).
                z = z - self._eta(t) * m * torch.sigmoid(z) * r
            elif self.variant == "learned_project":
                e = self._normalized_region_error(y, b_region, regions)
                sparse = center_scatter(e, regions.boxes, h, w)
                residual_fields.append(sparse)
                assert self.learned_projector is not None
                dz = self.learned_projector(f, y, sparse)
                z = z - self._eta(t) * dz
            else:
                raise RuntimeError(f"Unknown variant {self.variant}")
            y = F.softplus(z)
            iterates.append(y)

        out["y"] = y
        out["iterates"] = iterates
        out["residual_fields"] = residual_fields
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```


## File: `rmr_count/losses.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .operators import RegionSet, regional_sum


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions.

    This is deliberately a simple shared carrier loss, not a paper contribution.
    """
    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    pos = target > 0
    neg = ~pos
    terms = []
    if pos.any():
        terms.append(per[pos].mean())
    if neg.any():
        terms.append(per[neg].mean())
    if not terms:
        return per.mean()
    return torch.stack(terms).mean()


def global_count_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable global count loss on log1p counts."""
    pn = pred.sum(dim=(-2, -1))
    tn = target.sum(dim=(-2, -1))
    return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)


def scale_balanced_region_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 1.0,
) -> torch.Tensor:
    """Average region-count SmoothL1 equally across region scales."""
    if pred_region.shape != target_region.shape:
        raise ValueError(f"shape mismatch: {pred_region.shape} vs {target_region.shape}")
    losses = []
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        if mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_region[..., mask],
                    target_region[..., mask],
                    reduction="mean",
                    beta=beta,
                )
            )
    return torch.stack(losses).mean()


@dataclass
class LossConfig:
    lambda_global: float = 0.10
    lambda_region_map: float = 0.20
    lambda_region_head: float = 0.20
    lambda_deep_supervision: float = 0.10
    cell_beta: float = 1.0
    region_beta: float = 2.0


def compute_losses(
    outputs: dict,
    target_y: torch.Tensor,
    variant: str,
    cfg: LossConfig = LossConfig(),
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants.

    Variant semantics:
      direct:          fine + global only
      region_loss:     direct + training-only regional loss on final map
      region_aux:      direct + auxiliary regional evidence head
      learned_project: region_aux + learned inference projector
      rmr:             region_aux + exact-adjoint reconciliation
    """
    y = outputs["y"]
    regions: RegionSet = outputs["regions"]
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["global"] = global_count_loss(y, target_y)

    target_region = regional_sum(target_y, regions.boxes)

    if variant == "region_loss":
        pred_region = regional_sum(y, regions.boxes)
        losses["region_map"] = scale_balanced_region_loss(
            pred_region, target_region, regions, beta=cfg.region_beta
        )

    if variant in {"region_aux", "learned_project", "rmr"}:
        b_region = outputs["b_region"]
        losses["region_head"] = scale_balanced_region_loss(
            b_region, target_region, regions, beta=cfg.region_beta
        )

    # Optional weak deep supervision on intermediate positive measures for iterative variants.
    iterates = outputs.get("iterates", [])
    if variant in {"learned_project", "rmr"} and len(iterates) > 2:
        mids = iterates[1:-1]
        if mids:
            losses["deep"] = torch.stack([
                balanced_smooth_l1(m, target_y, beta=cfg.cell_beta) for m in mids
            ]).mean()

    total = losses["cell"] + cfg.lambda_global * losses["global"]
    if "region_map" in losses:
        total = total + cfg.lambda_region_map * losses["region_map"]
    if "region_head" in losses:
        total = total + cfg.lambda_region_head * losses["region_head"]
    if "deep" in losses:
        total = total + cfg.lambda_deep_supervision * losses["deep"]
    losses["total"] = total
    return losses
```


## File: `rmr_count/data.py`

```python
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def rasterize_points(
    points_xy: torch.Tensor,
    image_h: int,
    image_w: int,
    stride: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact stride-cell counts from point annotations.

    Canonical assignment:
        i = floor((y + 0.5) / stride)
        j = floor((x + 0.5) / stride)
    Points outside the actual image support are ignored, never clipped into a border cell.
    """
    gh = math.ceil(image_h / stride)
    gw = math.ceil(image_w / stride)
    out = torch.zeros((1, gh, gw), dtype=dtype)
    if points_xy.numel() == 0:
        return out

    pts = points_xy.float()
    x, y = pts[:, 0], pts[:, 1]
    valid = (x >= 0) & (x < image_w) & (y >= 0) & (y < image_h)
    if not valid.any():
        return out
    x, y = x[valid], y[valid]
    j = torch.floor((x + 0.5) / stride).long()
    i = torch.floor((y + 0.5) / stride).long()
    valid_cell = (i >= 0) & (i < gh) & (j >= 0) & (j < gw)
    i, j = i[valid_cell], j[valid_cell]
    flat = i * gw + j
    out.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return out


def _pad_to_crop(image: torch.Tensor, points: torch.Tensor, crop_h: int, crop_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = image.shape
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)
    if pad_h or pad_w:
        # ImageNet-normalized zero is close to mean after normalization; raw tensor here uses 0..1.
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    return image, points


def train_transform(
    image: Image.Image,
    points_xy: torch.Tensor,
    crop_size: int = 512,
    scale_range: tuple[float, float] = (0.75, 1.25),
    hflip_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometric augmentation that keeps point coordinates exact."""
    image_t = TF.to_tensor(image)
    pts = points_xy.clone().float()

    scale = random.uniform(*scale_range)
    h0, w0 = image_t.shape[-2:]
    h1 = max(32, int(round(h0 * scale)))
    w1 = max(32, int(round(w0 * scale)))
    image_t = TF.resize(image_t, [h1, w1], interpolation=InterpolationMode.BILINEAR, antialias=True)
    if pts.numel():
        pts[:, 0] *= w1 / w0
        pts[:, 1] *= h1 / h0

    image_t, pts = _pad_to_crop(image_t, pts, crop_size, crop_size)
    _, h, w = image_t.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    image_t = image_t[:, top:top + crop_size, left:left + crop_size]
    if pts.numel():
        pts[:, 0] -= left
        pts[:, 1] -= top
        keep = (
            (pts[:, 0] >= 0) & (pts[:, 0] < crop_size) &
            (pts[:, 1] >= 0) & (pts[:, 1] < crop_size)
        )
        pts = pts[keep]

    if random.random() < hflip_prob:
        image_t = torch.flip(image_t, dims=[-1])
        if pts.numel():
            pts[:, 0] = (crop_size - 1) - pts[:, 0]

    # Lightweight photometric augmentation.
    if random.random() < 0.5:
        image_t = TF.adjust_brightness(image_t, random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        image_t = TF.adjust_contrast(image_t, random.uniform(0.85, 1.15))

    return image_t.clamp(0, 1), pts


def normalize_image(image_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_t.dtype, device=image_t.device).view(3, 1, 1)
    return (image_t - mean) / std


class CrowdManifestDataset(Dataset):
    """Dataset over a standardized JSONL manifest.

    Each line:
      {"image": "relative/or/absolute/path.jpg", "points": [[x,y], ...], "id": "optional"}
    """

    def __init__(
        self,
        manifest: str | Path,
        train: bool,
        output_stride: int = 4,
        crop_size: int = 512,
        scale_range: tuple[float, float] = (0.75, 1.25),
    ):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.train = train
        self.output_stride = int(output_stride)
        self.crop_size = int(crop_size)
        self.scale_range = scale_range
        with self.manifest.open("r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        path = Path(item["image"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        pts = torch.tensor(item.get("points", []), dtype=torch.float32).reshape(-1, 2)

        if self.train:
            image_t, pts = train_transform(
                image, pts,
                crop_size=self.crop_size,
                scale_range=self.scale_range,
            )
        else:
            image_t = TF.to_tensor(image)

        h, w = image_t.shape[-2:]
        target_y = rasterize_points(pts, h, w, stride=self.output_stride)
        image_t = normalize_image(image_t)
        return {
            "image": image_t,
            "target_y": target_y,
            "points": pts,
            "id": item.get("id", path.stem),
            "path": str(path),
            "height": h,
            "width": w,
        }


def collate_train(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], 0),
        "target_y": torch.stack([b["target_y"] for b in batch], 0),
        "id": [b["id"] for b in batch],
    }


def collate_eval(batch: list[dict]) -> list[dict]:
    # Full-resolution images may differ in shape; evaluate sample-by-sample.
    return batch
```


## File: `rmr_count/prepare_manifest.py`

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _extract_points_mat(path: Path) -> np.ndarray:
    mat = loadmat(path)
    if "annPoints" in mat:
        pts = np.asarray(mat["annPoints"], dtype=np.float32)
        return pts.reshape(-1, 2)
    if "image_info" in mat:  # ShanghaiTech format
        pts = np.asarray(mat["image_info"][0, 0][0, 0][0], dtype=np.float32)
        return pts.reshape(-1, 2)
    # Conservative fallback: only accept an obvious Nx2 numeric array.
    candidates = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        a = np.asarray(v)
        if np.issubdtype(a.dtype, np.number) and a.ndim == 2 and a.shape[1] == 2:
            candidates.append((k, a))
    if len(candidates) == 1:
        return candidates[0][1].astype(np.float32)
    raise RuntimeError(f"Could not uniquely identify Nx2 points in {path}; keys={list(mat.keys())}")


def annotation_for(image: Path, ann_dir: Path, dataset: str) -> Path:
    stem = image.stem
    candidates: list[Path] = []
    if dataset.startswith("sha"):
        candidates += [ann_dir / f"GT_{stem}.mat", ann_dir / f"{stem}.mat"]
    elif dataset == "qnrf":
        candidates += [ann_dir / f"{stem}_ann.mat", ann_dir / f"{stem}.mat"]
    elif dataset == "nwpu":
        candidates += [ann_dir / f"{stem}.mat", ann_dir / f"{stem}_ann.mat"]
    else:
        candidates += [ann_dir / f"{stem}.mat"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No annotation for {image}; tried {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--dataset", choices=["sha_a", "sha_b", "qnrf", "nwpu"], required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    images = sorted([p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for image in images:
            ann = annotation_for(image, args.annotations, args.dataset)
            pts = _extract_points_mat(ann)
            row = {"image": str(image.resolve()), "points": pts.tolist(), "id": image.stem}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(images)} samples -> {args.out}")


if __name__ == "__main__":
    main()
```


## File: `rmr_count/split_manifest.py`

```python
from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--train-out", required=True, type=Path)
    ap.add_argument("--val-out", required=True, type=Path)
    ap.add_argument("--val-count", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    lines = [x for x in args.manifest.read_text().splitlines() if x.strip()]
    idx = list(range(len(lines)))
    random.Random(args.seed).shuffle(idx)
    n_val = args.val_count if args.val_count is not None else max(1, round(len(lines) * args.val_fraction))
    val_idx = set(idx[:n_val])
    train = [line for i, line in enumerate(lines) if i not in val_idx]
    val = [line for i, line in enumerate(lines) if i in val_idx]
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_text("\n".join(train) + "\n")
    args.val_out.write_text("\n".join(val) + "\n")
    print(f"total={len(lines)} train={len(train)} val={len(val)} seed={args.seed}")


if __name__ == "__main__":
    main()
```


## File: `rmr_count/metrics.py`

```python
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .operators import RegionSet, regional_sum


def count_from_map(y: torch.Tensor) -> torch.Tensor:
    return y.sum(dim=(-2, -1))


def game_single(pred: torch.Tensor, target: torch.Tensor, level: int) -> float:
    """Mass-preserving GAME(L) on one [1,H,W] count map."""
    if pred.ndim == 3:
        pred = pred[0]
    if target.ndim == 3:
        target = target[0]
    h, w = pred.shape
    n = 2 ** level
    ys = [round(i * h / n) for i in range(n + 1)]
    xs = [round(i * w / n) for i in range(n + 1)]
    err = 0.0
    for iy in range(n):
        for ix in range(n):
            p = pred[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]].sum()
            t = target[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]].sum()
            err += float((p - t).abs().item())
    return err


def summarize_predictions(rows: list[dict]) -> dict[str, float]:
    gt = np.asarray([r["gt"] for r in rows], dtype=np.float64)
    pred = np.asarray([r["pred"] for r in rows], dtype=np.float64)
    ae = np.abs(pred - gt)
    out = {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "NAE": float(np.mean(ae / np.maximum(gt, 1.0))),
        "Bias": float(np.mean(pred - gt)),
        "MedianAE": float(np.median(ae)),
        "P90AE": float(np.quantile(ae, 0.90)),
        "P95AE": float(np.quantile(ae, 0.95)),
        "MaxAE": float(ae.max(initial=0.0)),
    }
    for level in range(4):
        key = f"GAME{level}"
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = statistic(values[idx])
    lo = np.quantile(stats, alpha / 2)
    hi = np.quantile(stats, 1 - alpha / 2)
    return float(lo), float(hi)


def density_stratified_mae(rows: list[dict]) -> dict[str, float]:
    bins = {
        "sparse_le100": lambda n: n <= 100,
        "mid_101_500": lambda n: 100 < n <= 500,
        "dense_gt500": lambda n: n > 500,
    }
    out = {}
    for name, fn in bins.items():
        vals = [abs(r["pred"] - r["gt"]) for r in rows if fn(r["gt"])]
        if vals:
            out[name] = float(np.mean(vals))
    return out
```


## File: `rmr_count/train.py`

```python
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import CrowdManifestDataset, collate_eval, collate_train
from .losses import LossConfig, compute_losses
from .metrics import game_single, summarize_predictions
from .model import RMRConfig, RMRCount, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: dict) -> RMRCount:
    mcfg = RMRConfig(
        output_stride=cfg["model"].get("output_stride", 4),
        feature_width=cfg["model"].get("feature_width", 32),
        region_sizes_px=tuple(cfg["model"].get("region_sizes_px", [16, 32, 64, 128])),
        region_overlap=cfg["model"].get("region_overlap", 0.5),
        include_full_image=cfg["model"].get("include_full_image", True),
        iterations=cfg["model"].get("iterations", 2),
        eta_max=cfg["model"].get("eta_max", 1.0),
    )
    return RMRCount(mcfg, variant=cfg["model"]["variant"])


def make_loss_cfg(cfg: dict) -> LossConfig:
    x = cfg.get("loss", {})
    return LossConfig(
        lambda_global=x.get("lambda_global", 0.10),
        lambda_region_map=x.get("lambda_region_map", 0.20),
        lambda_region_head=x.get("lambda_region_head", 0.20),
        lambda_deep_supervision=x.get("lambda_deep_supervision", 0.10),
        cell_beta=x.get("cell_beta", 1.0),
        region_beta=x.get("region_beta", 2.0),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup: int):
    def fn(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / max(1, warmup))
        p = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)


@torch.no_grad()
def evaluate(model: RMRCount, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target_y"].to(device)
            out = model(image)
            y = out["y"][0]
            pred = float(y.sum().item())
            gt = float(target.sum().item())
            row = {"gt": gt, "pred": pred}
            for level in range(4):
                row[f"GAME{level}"] = game_single(y, target, level)
            rows.append(row)
    return summarize_predictions(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.lr is not None:
        cfg.setdefault("train", {})["lr"] = args.lr
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    train_ds = CrowdManifestDataset(
        cfg["data"]["train_manifest"],
        train=True,
        output_stride=cfg["model"].get("output_stride", 4),
        crop_size=cfg["data"].get("crop_size", 512),
        scale_range=tuple(cfg["data"].get("scale_range", [0.75, 1.25])),
    )
    val_manifest = cfg["data"].get("val_manifest")
    val_ds = None if not val_manifest else CrowdManifestDataset(
        val_manifest,
        train=False,
        output_stride=cfg["model"].get("output_stride", 4),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=cfg["train"].get("workers", 4),
        pin_memory=True,
        persistent_workers=cfg["train"].get("workers", 4) > 0,
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = None if val_ds is None else DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, cfg["train"].get("workers", 4))),
        collate_fn=collate_eval,
    )

    model = make_model(cfg).to(device)
    print(f"variant={model.variant} params={count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 1000))
    scheduler = make_scheduler(optimizer, epochs, int(cfg["train"].get("warmup_epochs", 25)))
    amp = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_cfg = make_loss_cfg(cfg)
    grad_clip = float(cfg["train"].get("grad_clip", 5.0))
    eval_every = int(cfg["train"].get("eval_every", 10))

    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", best_mae)

    log_path = out_dir / "train_log.csv"
    fieldnames = ["epoch", "lr", "train_total", "train_cell", "train_global", "clip_rate", "val_MAE", "val_RMSE", "val_NAE", "val_Bias"]
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, epochs):
        model.train()
        sums = {"total": 0.0, "cell": 0.0, "global": 0.0}
        n_steps = 0
        clipped = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(image)
                losses = compute_losses(outputs, target, model.variant, loss_cfg)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            clipped += int(float(grad_norm) > grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for k in sums:
                if k in losses:
                    sums[k] += float(losses[k].detach().item())
            n_steps += 1
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total": sums["total"] / max(1, n_steps),
            "train_cell": sums["cell"] / max(1, n_steps),
            "train_global": sums["global"] / max(1, n_steps),
            "clip_rate": clipped / max(1, n_steps),
            "val_MAE": "",
            "val_RMSE": "",
            "val_NAE": "",
            "val_Bias": "",
        }

        do_eval = val_loader is not None and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae": best_mae,
            "config": cfg,
        }
        if do_eval:
            metrics = evaluate(model, val_loader, device)
            row.update({
                "val_MAE": metrics["MAE"],
                "val_RMSE": metrics["RMSE"],
                "val_NAE": metrics["NAE"],
                "val_Bias": metrics["Bias"],
            })
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                state["best_mae"] = best_mae
                torch.save(state, out_dir / "best_val_mae.pt")
            print(f"ep={epoch:04d} loss={row['train_total']:.4f} valMAE={metrics['MAE']:.3f} valRMSE={metrics['RMSE']:.3f} clip={row['clip_rate']:.3f}")
        else:
            print(f"ep={epoch:04d} loss={row['train_total']:.4f} clip={row['clip_rate']:.3f}")
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            state["best_mae"] = best_mae
            torch.save(state, out_dir / "last.pt")

        with log_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


if __name__ == "__main__":
    main()
```


## File: `rmr_count/eval.py`

```python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import CrowdManifestDataset, collate_eval
from .metrics import bootstrap_ci, density_stratified_mae, game_single, summarize_predictions
from .model import RMRConfig, RMRCount
from .operators import regional_sum


def make_model_from_ckpt(ckpt: dict, device: torch.device) -> RMRCount:
    cfg = ckpt["config"]
    mcfg = RMRConfig(
        output_stride=cfg["model"].get("output_stride", 4),
        feature_width=cfg["model"].get("feature_width", 32),
        region_sizes_px=tuple(cfg["model"].get("region_sizes_px", [16, 32, 64, 128])),
        region_overlap=cfg["model"].get("region_overlap", 0.5),
        include_full_image=cfg["model"].get("include_full_image", True),
        iterations=cfg["model"].get("iterations", 2),
        eta_max=cfg["model"].get("eta_max", 1.0),
    )
    model = RMRCount(mcfg, variant=cfg["model"]["variant"])
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_direct(model: RMRCount, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    out = model(image.unsqueeze(0))
    return out["y"][0], out


def _aligned_floor(v: int, stride: int) -> int:
    return (v // stride) * stride


def _aligned_ceil(v: int, stride: int) -> int:
    return ((v + stride - 1) // stride) * stride


@torch.no_grad()
def predict_tiled(
    model: RMRCount,
    image: torch.Tensor,
    tile_size: int = 512,
    halo: int = 0,
) -> torch.Tensor:
    """Core/halo tiled prediction assembled without double-counting.

    Core boundaries are aligned to output stride except the final image boundary.
    Halo affects context only; only the core prediction is written to the output.
    """
    _, h, w = image.shape
    s = model.cfg.output_stride
    tile_size = max(s, _aligned_floor(tile_size, s))
    halo = max(0, _aligned_floor(halo, s))
    gh, gw = math.ceil(h / s), math.ceil(w / s)
    canvas = image.new_zeros((1, gh, gw))

    ys = list(range(0, h, tile_size))
    xs = list(range(0, w, tile_size))
    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)

            sy0 = max(0, _aligned_floor(y0 - halo, s))
            sx0 = max(0, _aligned_floor(x0 - halo, s))
            sy1 = min(h, _aligned_ceil(y1 + halo, s))
            sx1 = min(w, _aligned_ceil(x1 + halo, s))
            patch = image[:, sy0:sy1, sx0:sx1].unsqueeze(0)
            y_patch = model(patch)["y"][0]

            gy0 = y0 // s
            gx0 = x0 // s
            gy1 = math.ceil(y1 / s)
            gx1 = math.ceil(x1 / s)
            ly0 = (y0 - sy0) // s
            lx0 = (x0 - sx0) // s
            hh = gy1 - gy0
            ww = gx1 - gx0
            canvas[:, gy0:gy1, gx0:gx1] = y_patch[:, ly0:ly0 + hh, lx0:lx0 + ww]
    return canvas


@torch.no_grad()
def evaluate(
    model: RMRCount,
    loader: DataLoader,
    device: torch.device,
    tile_size: int,
    practical_halo: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    region_errors: dict[int, list[float]] = defaultdict(list)

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].to(device)
            target = sample["target_y"].to(device)
            y, out = predict_direct(model, image)
            y_t0 = predict_tiled(model, image, tile_size=tile_size, halo=0)
            y_th = predict_tiled(model, image, tile_size=tile_size, halo=practical_halo)

            gt = float(target.sum().item())
            pred = float(y.sum().item())
            pred_t0 = float(y_t0.sum().item())
            pred_th = float(y_th.sum().item())
            row = {
                "id": sample["id"],
                "gt": gt,
                "pred": pred,
                "pred_tiled_h0": pred_t0,
                "pred_tiled_practical": pred_th,
                "abs_err": abs(pred - gt),
                "direct_tiled_h0_abs": abs(pred - pred_t0),
                "direct_tiled_practical_abs": abs(pred - pred_th),
                "direct_tiled_h0_norm": abs(pred - pred_t0) / max(gt, 1.0),
                "direct_tiled_practical_norm": abs(pred - pred_th) / max(gt, 1.0),
            }
            for level in range(4):
                row[f"GAME{level}"] = game_single(y, target, level)
            rows.append(row)

            regions = out["regions"]
            p_reg = regional_sum(y.unsqueeze(0), regions.boxes)[0, 0]
            t_reg = regional_sum(target.unsqueeze(0), regions.boxes)[0, 0]
            ae = (p_reg - t_reg).abs()
            for sid in torch.unique(regions.scale_id):
                m = regions.scale_id == sid
                region_errors[int(sid.item())].extend(ae[m].detach().cpu().tolist())

    summary = summarize_predictions(rows)
    summary.update(density_stratified_mae(rows))
    summary["DirectTiledH0_MeanAbs"] = float(np.mean([r["direct_tiled_h0_abs"] for r in rows]))
    summary["DirectTiledH0_MeanNorm"] = float(np.mean([r["direct_tiled_h0_norm"] for r in rows]))
    summary["DirectTiledPractical_MeanAbs"] = float(np.mean([r["direct_tiled_practical_abs"] for r in rows]))
    summary["DirectTiledPractical_MeanNorm"] = float(np.mean([r["direct_tiled_practical_norm"] for r in rows]))

    paired = np.asarray([r["direct_tiled_practical_norm"] for r in rows], dtype=np.float64)
    lo, hi = bootstrap_ci(paired, n_boot=5000)
    summary["DirectTiledPractical_MeanNorm_CI95_lo"] = lo
    summary["DirectTiledPractical_MeanNorm_CI95_hi"] = hi

    for sid, vals in region_errors.items():
        name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
        summary[f"RegionMAE_px_{name}"] = float(np.mean(vals)) if vals else float("nan")
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--practical-halo", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = make_model_from_ckpt(ckpt, device)
    ds = CrowdManifestDataset(args.manifest, train=False, output_stride=model.cfg.output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_eval)

    rows, summary = evaluate(model, loader, device, args.tile_size, args.practical_halo)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```


## File: `rmr_count/profile.py`

```python
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .model import RMRConfig, RMRCount, count_parameters


@torch.no_grad()
def profile_latency(model: torch.nn.Module, x: torch.Tensor, warmup: int = 100, iters: int = 500) -> dict:
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {
        "latency_ms_mean": float(a.mean()),
        "latency_ms_p50": float(np.quantile(a, 0.50)),
        "latency_ms_p95": float(np.quantile(a, 0.95)),
        "fps_from_mean": float(1000.0 / a.mean()),
    }


def profiler_flops(model: torch.nn.Module, x: torch.Tensor) -> float | None:
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if x.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(x)
        total = sum((evt.flops or 0) for evt in prof.key_averages())
        return float(total)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rmr", choices=["direct", "region_loss", "region_aux", "learned_project", "rmr"])
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RMRCount(RMRConfig(iterations=args.iterations), variant=args.variant).to(device).eval()
    x = torch.randn(1, 3, args.height, args.width, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = {
        "variant": args.variant,
        "iterations": args.iterations,
        "params": count_parameters(model),
        "input": [1, 3, args.height, args.width],
    }
    result.update(profile_latency(model, x))
    result["profiler_flops"] = profiler_flops(model, x)
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```


## File: `rmr_count/aggregate.py`

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    args = ap.parse_args()
    rows = [json.loads(Path(p).read_text()) for p in args.summaries]
    keys = sorted(set.intersection(*(set(r) for r in rows)))
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            a = np.asarray(vals, dtype=np.float64)
            out[k] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "n": len(a),
            }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```


## File: `configs/rmr_t2.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/rmr_t2_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: rmr
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `configs/rmr_t1.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/rmr_t1_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: rmr
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 1
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `configs/direct.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/direct_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: direct
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `configs/region_loss.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/region_loss_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: region_loss
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `configs/region_aux.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/region_aux_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: region_aux
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `configs/learned_project.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/learned_project_seed42

data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range: [0.75, 1.25]

model:
  variant: learned_project
  output_stride: 4
  feature_width: 32
  region_sizes_px: [16, 32, 64, 128]
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 1.0

loss:
  lambda_global: 0.10
  lambda_region_map: 0.20
  lambda_region_head: 0.20
  lambda_deep_supervision: 0.10
  cell_beta: 1.0
  region_beta: 2.0

train:
  batch_size: 8
  workers: 4
  lr: 0.0003
  weight_decay: 0.0001
  epochs: 1000
  warmup_epochs: 25
  eval_every: 10
  grad_clip: 5.0
  amp: true
```


## File: `run_matrix.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Pilot LR sweep on validation only. Freeze the selected LR before multi-seed final runs.
for lr in 1e-4 3e-4 1e-3; do
  python -m rmr_count.train \
    --config configs/rmr_t2.yaml \
    --seed 42 \
    --lr "$lr" \
    --output-dir "runs/sha_a/lr_sweep_rmr_t2_${lr}"
done

# Matched RQ matrix after choosing LR using validation only.
LR=3e-4   # replace only with the validation-selected value
for seed in 42 123 3407; do
  for cfg in direct region_loss region_aux learned_project rmr_t1 rmr_t2; do
    python -m rmr_count.train \
      --config "configs/${cfg}.yaml" \
      --seed "$seed" \
      --lr "$LR" \
      --output-dir "runs/sha_a/${cfg}_seed${seed}"
  done
done
```


## File: `tests/test_operators.py`

```python
import torch

from rmr_count.operators import (
    build_multiscale_regions,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_adjoint,
    regional_sum,
)


def test_rectangle_sum_matches_naive():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 11, 13, dtype=torch.float64)
    boxes = torch.tensor([[0, 0, 3, 4], [2, 5, 11, 13], [7, 1, 10, 8]], dtype=torch.long)
    got = regional_sum(x, boxes)
    want = []
    for y1, x1, y2, x2 in boxes.tolist():
        want.append(x[..., y1:y2, x1:x2].sum(dim=(-2, -1)))
    want = torch.stack(want, dim=-1)
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_adjoint_identity():
    torch.manual_seed(1)
    b, c, h, w = 2, 2, 12, 15
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    x = torch.randn(b, c, h, w, dtype=torch.float64)
    e = torch.randn(b, c, regions.boxes.shape[0], dtype=torch.float64)
    ax = regional_sum(x, regions.boxes)
    ate = regional_adjoint(e, regions.boxes, h, w)
    lhs = (ax * e).sum()
    rhs = (x * ate).sum()
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_full_image_region_present():
    r = build_multiscale_regions(9, 10, output_stride=4, region_sizes_px=(16,), include_full_image=True)
    assert any(tuple(b.tolist()) == (0, 0, 9, 10) for b in r.boxes)
```


## File: `tests/test_model.py`

```python
import torch

from rmr_count.model import RMRConfig, RMRCount
from rmr_count.operators import regional_sum


def test_rmr_output_positive_and_shape():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="rmr")
    x = torch.randn(2, 3, 128, 160)
    out = model(x)
    y = out["y"]
    assert y.shape == (2, 1, 32, 40)
    assert torch.all(y >= 0)
    assert len(out["iterates"]) == 3


def test_zero_region_residual_is_fixed_direction():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=1), variant="rmr")
    y = torch.rand(1, 1, 16, 20)
    regions = model._regions(16, 20, y.device)
    b = regional_sum(y, regions.boxes)
    r = model._rmr_field(y, b, regions)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)
```


## File: `tests/test_data.py`

```python
import torch

from rmr_count.data import rasterize_points


def test_rasterize_points_conserves_count():
    pts = torch.tensor([[0.0, 0.0], [3.6, 4.0], [7.4, 7.4], [15.0, 15.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 4


def test_oob_points_are_ignored_not_clipped():
    pts = torch.tensor([[-1.0, 2.0], [2.0, 2.0], [20.0, 3.0]])
    y = rasterize_points(pts, 16, 16, stride=4)
    assert y.sum().item() == 1
```


# 64. Dataset preparation and split protocol

## 64.1 Standardized manifest

All datasets are converted once into JSONL rows:

```json
{"image":"/abs/path/IMG_1.jpg","points":[[x1,y1],[x2,y2]],"id":"IMG_1"}
```

The model never trains on a Gaussian pseudo-density target. The canonical target is an exact stride-cell count map constructed directly from the point annotations.

### ShanghaiTech A example

```bash
python -m rmr_count.prepare_manifest \
  --dataset sha_a \
  --images /data/ShanghaiTech/part_A/train_data/images \
  --annotations /data/ShanghaiTech/part_A/train_data/ground-truth \
  --out data/sha_a_all_train.jsonl
```

Create the frozen 270/30 development split:

```bash
python -m rmr_count.split_manifest \
  --manifest data/sha_a_all_train.jsonl \
  --train-out data/sha_a_train.jsonl \
  --val-out data/sha_a_val.jsonl \
  --val-count 30 \
  --seed 2026
```

Generate the official test manifest separately. **Never use the official test split for hyperparameter or checkpoint selection.**

## 64.2 QNRF and NWPU

Use the same standardized manifest. For QNRF, create a deterministic validation subset only from the official training set. For NWPU, use its official validation set for development and the official server/test protocol for final reporting.

## 64.3 Final retraining rule

After all design choices are frozen on validation:

1. choose the architecture, LR, region set, iteration count, and fixed training epoch budget;
2. retrain on the full official training set;
3. evaluate the frozen final model on test exactly once per seed/protocol;
4. do not select a “best test checkpoint.”

---

# 65. Training commands

## 65.1 Unit tests first

```bash
PYTHONPATH=. pytest -q tests
```

Expected for the reference package:

```text
7 passed
```

The most important test is the numerical adjoint identity:

\[
\langle AY,e\rangle=\langle Y,A^\top e\rangle.
\]

If this test fails, no experiment is scientifically valid.

## 65.2 Pilot LR sweep

Registered sweep:

\[
\boxed{\{10^{-4},3\times10^{-4},10^{-3}\}}
\]

Run only on validation:

```bash
for lr in 1e-4 3e-4 1e-3; do
  python -m rmr_count.train \
    --config configs/rmr_t2.yaml \
    --seed 42 \
    --lr $lr \
    --output-dir runs/sha_a/lr_sweep_${lr}
done
```

Choose the LR using validation only, then freeze it for all matched variants.

## 65.3 Full causal matrix

```bash
bash run_matrix.sh
```

Final claims require at least three seeds:

```text
42, 123, 3407
```

All matched variants use the same:

- crop size;
- augmentation;
- optimizer;
- LR;
- weight decay;
- epoch budget;
- validation rule;
- output stride;
- feature width;
- base fine-measure loss.

---

# 66. Evaluation commands

Direct + controlled tiled + practical tiled + GAME + regional diagnostics:

```bash
python -m rmr_count.eval \
  --checkpoint runs/sha_a/rmr_t2_seed42/best_val_mae.pt \
  --manifest data/sha_a_test.jsonl \
  --out-dir eval/sha_a/rmr_t2_seed42 \
  --tile-size 512 \
  --practical-halo 64
```

Outputs:

```text
eval/.../predictions.csv
eval/.../summary.json
```

The evaluator reports:

- MAE;
- RMSE;
- NAE;
- Bias;
- Median/P90/P95/Max absolute error;
- GAME(0–3);
- sparse/mid/dense count strata;
- region-scale MAE;
- direct vs controlled-tiled discrepancy;
- direct vs halo-tiled discrepancy;
- normalized paired direct/tiled discrepancy;
- bootstrap 95% CI for the normalized paired discrepancy.

## 66.1 Efficiency profile

```bash
python -m rmr_count.profile \
  --variant rmr \
  --iterations 2 \
  --height 512 \
  --width 512 \
  --device cuda
```

Record:

- trainable parameters;
- profiler FLOPs, with the profiler convention stated;
- mean/p50/p95 batch-1 latency;
- FPS;
- peak allocated GPU memory;
- GPU model;
- PyTorch/CUDA version;
- FP32 and AMP separately if both are reported.

Do not compare latency numbers measured on different GPUs as if they were directly equivalent.

---

# 67. Required experiment table before any CVPR claim

| Variant | Region supervision | Region evidence at inference | Projection | Params | FLOPs | Latency | SHA MAE | SHA RMSE |
|---|---|---|---|---:|---:|---:|---:|---:|
| B0 Direct | no | no | none | | | | | |
| B1 RegionLoss | yes | no | none | | | | | |
| B2 RegionAux | yes | yes | none | | | | | |
| B3 LearnedProject | yes | yes | learned | | | | | |
| B4 RMR-T1 | yes | yes | exact \(A^\top\) | | | | | |
| B5 RMR-T2 | yes | yes | exact \(A^\top\) | | | | | |

The paper mechanism survives only if, with repeated seeds:

\[
\boxed{B5 < B2}
\]

and ideally:

\[
\boxed{B5 < B3}
\]

for MAE/RMSE with a meaningful efficiency trade-off.

---

# 68. Statistical reporting

For the final ablation table report, for every major variant:

\[
\text{mean}\pm\text{std}
\]

over at least three seeds.

For paired per-image comparisons between B3 and B5, use bootstrap confidence intervals on:

\[
\Delta_i = |\hat N_i^{B5}-N_i|-|\hat N_i^{B3}-N_i|.
\]

Report:

- mean paired MAE difference;
- 95% bootstrap CI;
- fraction of images improved;
- density-stratified paired difference.

Do not rely only on independently selected best MAEs from different checkpoints.

---

# 69. Exact RQ acceptance criteria

## RQ1

**Question:** under an ultra-light budget, does exact count-operator reasoning outperform spending similar capacity on learned region-to-grid correction?

Primary test:

\[
B5\text{ (RMR)}\quad\text{vs}\quad B3\text{ (LearnedProject)}.
\]

Accept H1 only if RMR has a repeated-seed accuracy advantage without a worse deployment trade-off that nullifies it.

## RQ2

**Question:** does regional information help more when used during inference than only as a training signal/auxiliary task?

Primary tests:

\[
B5\text{ vs }B1,
\qquad
B5\text{ vs }B2.
\]

If B1 or B2 matches B5, the inference-time reconciliation claim fails.

---

# 70. Mandatory mechanism diagnostics

For each test image and each RMR iteration record:

\[
q^{(t)}=AY^{(t)},
\]

\[
e^{(t)}=q^{(t)}-b,
\]

and:

\[
r^{(t)}=
\frac{A^\top[(q^{(t)}-b)/|R|]}
{A^\top\mathbf 1+\epsilon}.
\]

Required plots:

1. regional residual norm before/after each iteration;
2. MAE versus iteration \(T=0,1,2,3\);
3. regional MAE versus physical region size;
4. GAME(0–3) before/after reconciliation;
5. error versus ground-truth count;
6. empty-region predicted mass;
7. latency overhead versus T;
8. qualitative examples showing \(Y^{(0)}\), regional residuals, \(r^{(t)}\), and \(Y^*\).

A mechanism plot is not allowed to substitute for benchmark MAE/RMSE.

---

# 71. Strongest additional controls if the core survives

Run these **only after B5 beats the registered controls**.

### Oracle regional evidence

Replace \(b_R\) with exact GT regional counts at evaluation as an upper-bound analysis. Never report this as normal inference.

### Shuffled regional evidence

Permute \(b_R\) among same-scale regions. A meaningful method should degrade.

### Region scale set

\[
\{32\},
\{32,64\},
\{32,64,128\},
\{16,32,64,128\}.
\]

### Overlap

Compare 0% vs 50% overlap. The 50% setting is expected to reduce arbitrary partition dependence; it must be demonstrated rather than asserted.

### Backbone independence

After the central mechanism survives on the tiny carrier, repeat B0/B3/B5 on a second compact carrier. This demonstrates that RMR is not an accident of one encoder.

---

# 72. Final paper checklist

Before writing the final CVPR submission, all boxes below must be satisfied.

## Scientific

- [ ] gap re-checked against 2026/current literature;
- [ ] RQ1 and RQ2 stated before method details;
- [ ] B0–B5 run with matched settings;
- [ ] B5 beats training-only regional use;
- [ ] B5 beats or clearly Pareto-dominates learned projection;
- [ ] at least three final seeds;
- [ ] no test-set model selection;
- [ ] no unsupported “first” claim;
- [ ] limitations and failure cases included.

## Accuracy

- [ ] SHA;
- [ ] QNRF;
- [ ] NWPU;
- [ ] SHB/JHU if space allows;
- [ ] current lightweight baselines;
- [ ] strong modern non-lightweight references for context;
- [ ] ZIP comparison where protocol is compatible;
- [ ] DM-Count/PML carrier comparison or discussion where relevant.

## Efficiency

- [ ] Params;
- [ ] FLOPs/MAC convention explicitly stated;
- [ ] batch-1 latency;
- [ ] p50/p95 latency;
- [ ] peak memory;
- [ ] named hardware;
- [ ] actual operator overhead measured.

## Reproducibility

- [ ] exact commit hash;
- [ ] configs in supplementary;
- [ ] seeds;
- [ ] split files/checksums;
- [ ] environment versions;
- [ ] code unit tests;
- [ ] evaluation script frozen before final runs.

---

# 73. Final paper-level framing

The paper should ultimately read as one argument:

> **Crowd counting needs fine local evidence, but ultra-light models have limited capacity for learned contextual machinery. Prior work already shows that regional counts are useful, primarily through supervision, aggregation, or hierarchical prediction. We therefore ask whether known counting algebra can supply part of regional reasoning directly at inference. RMR-Count predicts a fine non-negative measure and separately inferred overlapping regional counts, compares them through an exact regional-sum operator, and maps the discrepancy back through the exact adjoint before a tiny visual preconditioner allocates the correction. The decisive experiments compare this mechanism against training-only regional supervision and a stronger/larger learned projector under the same carrier and protocol.**

If the registered comparisons fail, the correct action is to reject/reframe RMR-Count rather than add unrelated modules.
