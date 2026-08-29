# NTPC: Neural Tree-Pólya Crowd Counting
## Full Research Specification, Mathematical Formulation, PyTorch Reference Implementation, and Experimental Protocol

**Status:** research specification  
**Task:** ultra-lightweight point-supervised crowd counting  
**Primary objective:** strong counting accuracy with a clearly defensible contribution under a sub-0.5M parameter budget  
**Training protocol:** from scratch; no old checkpoint, no teacher, no knowledge distillation, no inherited weights  
**Core idea:** learn one positive conserved mass map and train its regional allocation as an image-conditioned Tree-Pólya / Dirichlet-tree count process.

---

# 1. Research question

The central question is:

> **Can an ultra-lightweight crowd counter learn more accurate spatial count allocation from point annotations by modeling exact regional counts as a conditional probabilistic count tree, instead of independently regressing density/count targets?**

The proposed model predicts only one positive mass map

\[
D(I)\in\mathbb{R}_{+}^{H/4\times W/4}.
\]

Every global or regional predicted count is obtained by summing this same map.

The probabilistic training model factorizes crowd counts into:

1. **global magnitude**
2. **coarse spatial allocation**
3. **progressively finer conditional allocation**
4. **extra fine supervision only in genuinely dense regions**

The intended scientific claim is **not** that Dirichlet-Multinomial or Tree-Pólya distributions are new.

The intended contribution is:

> **an image-conditioned neural parameterization of hierarchical count splitting for point-supervised crowd counting, driven by one conserved mass map.**

---

# 2. What is and is not novel

## 2.1 Not claimed as novel

The following ideas already have strong prior art and must not be presented as standalone novelty:

- point annotations converted to exact regional integer counts;
- Gaussian-free crowd supervision;
- multiscale regional count supervision;
- local block counting;
- Poisson / zero-inflated probabilistic block counting;
- spatial divide-and-conquer;
- quadtree refinement in dense regions;
- generic hard-region mining;
- contrastive foreground/background learning;
- lightweight FPN/context modules.

Important related works include:

- **DM-Count**: distribution matching without Gaussian density-map targets.
- **S-DCNet / SS-DCNet**: spatial divide-and-conquer and count redistribution.
- **PET**: adaptive quadtree point querying in dense regions.
- **ZIP**: zero-inflated Poisson modeling for sparse block counts.
- **Dirichlet-Tree Multinomial**: established statistical model for over-dispersed tree-structured count data.
- **Tree Pólya Splitting**: general framework combining a total-count distribution with conditional count splitting along a partition tree.

## 2.2 Proposed contribution

The proposed contribution is the combination

\[
\boxed{
\text{one conserved neural mass map}
+
\text{exact point-derived node counts}
+
\text{probabilistic conditional count allocation}
+
\text{density-adaptive likelihood depth}
}
\]

with no Gaussian target map and no inference-time adaptive branching.

A safe paper claim is:

> We formulate point-supervised crowd counting as an image-conditioned hierarchical count-allocation problem. A single positive mass map determines both total crowd magnitude and conditional regional proportions, and exact point-derived counts supervise the resulting Dirichlet-tree likelihood.

Do **not** write “first Dirichlet-tree model ever” or “first hierarchical crowd counter”.

Prefer:

> To our knowledge, we did not find a prior crowd-counting method that parameterizes a Dirichlet-tree / Tree-Pólya count-allocation likelihood directly from a single conserved neural mass map.

---

# 3. Model identity

A concise identity for the method is

\[
\boxed{
\textbf{One conserved map, one probabilistic count tree.}
}
\]

The model itself may use any sufficiently strong ultra-lightweight feature extractor.

The learning formulation is intentionally architecture-agnostic.

At inference:

```text
Image
  ↓
Ultra-lightweight CNN / hybrid backbone
  ↓
Lightweight neck / FPN
  ↓
Mass head
  ↓
Softplus
  ↓
D >= 0
  ↓
predicted count = sum(D)
```

No tree traversal is required at inference.

No Dirichlet sampling is required.

No region router is required.

No auxiliary head is required.

---

# 4. Notation

Let an image be

\[
I\in\mathbb{R}^{3\times H\times W}.
\]

Its point annotation set is

\[
\mathcal P=
\{(x_i,y_i)\}_{i=1}^{N}.
\]

The true image count is

\[
N=|\mathcal P|.
\]

The network outputs a stride-4 positive mass map

\[
D=f_\theta(I),
\qquad
D\ge0,
\qquad
D\in\mathbb R^{H/4\times W/4}.
\]

In practice:

\[
D=\operatorname{Softplus}(Z)
\]

where \(Z\) is the raw one-channel mass logit map.

---

# 5. Exact point-derived regional targets

No Gaussian kernel is used. Point annotations \((x_i, y_i)\) lie in the continuous pixel support \([-0.5, W-0.5] \times [-0.5, H-0.5]\).

For a square block size \(B\in\{4, 8, 16, 32, 64\}\), pixel coordinates map to discrete grid cells via zero-based pixel centers:

\[
c_x^{(B)}(i)
=
\operatorname{clip}
\left(
\left\lfloor
\frac{x_i+0.5}{B}
\right\rfloor,
0,W_B-1
\right),
\]

\[
c_y^{(B)}(i)
=
\operatorname{clip}
\left(
\left\lfloor
\frac{y_i+0.5}{B}
\right\rfloor,
0,H_B-1
\right),
\]

where \(W_B = \lceil W/B \rceil, H_B = \lceil H/B \rceil\). The exact integer count target for cell \((u, v)\) is:

\[
Y^{(B)}_{u,v}
=
\sum_{i=1}^{N}
\mathbf 1
\left[
c_y^{(B)}(i)=u,\;
c_x^{(B)}(i)=v
\right].
\]

In practice, \(Y^4\) is rasterized as the authoritative leaf level. All coarser levels are generated recursively by exact \(2\times2\) summation:

\[
Y^4 \xrightarrow{\text{SumPool}_2} Y^8 \xrightarrow{\text{SumPool}_2} Y^{16} \xrightarrow{\text{SumPool}_2} Y^{32} \xrightarrow{\text{SumPool}_2} Y^{64} \rightarrow N.
\]

Exact conservation holds by construction across all levels:

\[
\sum_{u,v} Y^4_{u,v} = \sum_{u,v} Y^8_{u,v} = \sum_{u,v} Y^{16}_{u,v} = \sum_{u,v} Y^{32}_{u,v} = \sum_{u,v} Y^{64}_{u,v} = N.
\]

---

# 6. Predicted regional counts from one mass map

Because \(D\) is at stride 4, a \(B\times B\) image region corresponds to

\[
k_B=B/4
\]

cells per side in the mass map.

Define

\[
\mu^{(B)}
=
\operatorname{SumPool}_{k_B}(D).
\]

Equivalently,

\[
\mu^{(B)}
=
k_B^2
\operatorname{AvgPool}_{k_B}(D).
\]

Thus

\[
\mu^{16}_{u,v}
=
\sum_{(a,b)\in\{0,1,2,3\}^{2}}
D_{4u+a,4v+b}.
\]

The predicted global count is

\[
\hat N
=
\mu_N
=
\sum_{x,y} D_{x,y}.
\]

All regional counts come from the same \(D\).

Hence conservation is exact by construction:

\[
\mu^{64}_{p}
=
\sum_{c\in child(p)}\mu^{32}_{c},
\]

\[
\mu^{32}_{p}
=
\sum_{c\in child(p)}\mu^{16}_{c}.
\]

No explicit “conservation loss” is needed.

---

# 7. Root magnitude model

The spatial allocation likelihood is compositional; it mainly determines *where* mass should go.

Absolute magnitude is handled at the root.

Use a Negative Binomial model:

\[
N
\sim
NB(\mu_N,r)
\]

where

\[
\mu_N=\sum D.
\]

We use the mean-dispersion parameterization

\[
\operatorname{Var}(N)
=
\mu_N+\frac{\mu_N^2}{r}.
\]

As \(r\rightarrow\infty\), the model approaches Poisson dispersion.

The PMF is

\[
P(N=n)
=
\frac{\Gamma(n+r)}
{\Gamma(r)\Gamma(n+1)}
\left(
\frac{r}{r+\mu}
\right)^r
\left(
\frac{\mu}{r+\mu}
\right)^n.
\]

The negative log-likelihood is

\[
\mathcal L_{NB}
=
-
\log P(N\mid \mu_N,r).
\]

Official experiments use fixed \(r = 50.0\) for every matched ablation. The dispersion parameter \(r\) is bounded in \((0, 10000]\) and is not dynamically re-estimated across splits.

Root alternatives:
- **L1 loss**
- **Poisson NLL**
- **Negative-Binomial NLL**

Any change to \(r\) is treated as a separate ablation.

---

# 8. Conditional spatial allocation

For a parent region \(p\) with \(K\) children, let predicted positive child masses be

\[
\mu_{p,1},\ldots,\mu_{p,K}.
\]

Clamp them from below and normalize to obtain the composition:

\[
\tilde\mu_{p,c} = \max(\mu_{p,c}, \epsilon),
\qquad
\pi_{p,c}
=
\frac{\tilde\mu_{p,c}}
{\sum_{j=1}^{K}\tilde\mu_{p,j}}.
\]

Then

\[
\sum_{c=1}^{K}\pi_{p,c}=1.
\]

For a \(2\times2\) spatial split,

\[
K=4.
\]

The true child count vector is

\[
\mathbf Y_p=
(Y_{p,1},Y_{p,2},Y_{p,3},Y_{p,4})
\]

with parent count

\[
Y_p=\sum_cY_{p,c}.
\]

---

# 9. Deterministic allocation baseline

A strong baseline should match the same conservation structure without using a probabilistic over-dispersed likelihood.

Use the ground-truth parent magnitude and predicted proportions:

\[
\tilde Y_{p,c}
=
Y_p\pi_{p,c}.
\]

By construction,

\[
\sum_c\tilde Y_{p,c}=Y_p.
\]

A deterministic allocation loss is

\[
\mathcal L_{\text{Det}}
=
\frac{1}{|\mathcal V|}
\sum_{p\in\mathcal V}
\frac{
\sum_c
\left|
Y_{p,c}-Y_p\pi_{p,c}
\right|
}
{Y_p+\epsilon}.
\]

Only parents with \(Y_p>0\) contribute.

This baseline tests whether the gain comes simply from hierarchical count decomposition.

---

# 10. Multinomial allocation baseline

Another baseline is

\[
\mathbf Y_p
\mid Y_p
\sim
Multinomial(Y_p,\boldsymbol\pi_p).
\]

Ignoring target-only constants,

\[
-\log P(\mathbf Y_p\mid Y_p,\pi_p)
=
-\sum_cY_{p,c}\log\pi_{p,c}
+
C(Y).
\]

A hierarchy of pure multinomial splits is largely a refactorization of leaf probabilities, so it should be treated as a baseline rather than the main method.

---

# 11. Dirichlet-Multinomial allocation

To model overdispersion, define

\[
\alpha_{p,c}
=
\kappa_l\pi_{p,c}
\]

for tree level \(l\).

The concentration sum is

\[
\alpha_{p,0}
=
\sum_c\alpha_{p,c}
=
\kappa_l.
\]

Then

\[
\boxed{
\mathbf Y_p
\mid
Y_p,I
\sim
DirichletMultinomial
(Y_p,\boldsymbol\alpha_p)
}
\]

with PMF

\[
P(\mathbf y\mid n,\boldsymbol\alpha)
=
\frac{n!}{\prod_c y_c!}
\frac{\Gamma(\alpha_0)}
{\Gamma(n+\alpha_0)}
\prod_c
\frac{\Gamma(y_c+\alpha_c)}
{\Gamma(\alpha_c)}.
\]

The log-likelihood is

\[
\log P
=
\log\Gamma(n+1)
-
\sum_c\log\Gamma(y_c+1)
+
\log\Gamma(\alpha_0)
-
\log\Gamma(n+\alpha_0)
+
\sum_c
\left[
\log\Gamma(y_c+\alpha_c)
-
\log\Gamma(\alpha_c)
\right].
\]

The concentration \(\kappa_l\) controls dispersion.

Large \(\kappa_l\):

\[
DM\rightarrow Multinomial.
\]

Small \(\kappa_l\):

more overdispersion and stronger allowance for heterogeneous spatial allocations.

Recommended initial ablation:

\[
\kappa_l\in\{5,10,20,50,\infty\}.
\]

Here \(\infty\) is represented by the Multinomial baseline.

---

# 12. Neural Dirichlet-tree likelihood

The main model factorizes the exact spatial counts hierarchically.

Use:

```text
Global count N
      |
      v
all 64x64 blocks
      |
      v
each 64 -> four 32
      |
      v
each 32 -> four 16
      |
      v
dense 16 -> four 8
```

The base tree is

\[
N\rightarrow64\rightarrow32\rightarrow16.
\]

The likelihood is

\[
P(Y\mid I)
=
P(N\mid I)
P(Y^{64}\mid N,I)
\prod_{p\in64}
P(Y^{32}_{child(p)}\mid Y^{64}_p,I)
\prod_{p\in32}
P(Y^{16}_{child(p)}\mid Y^{32}_p,I).
\]

The root term is Negative Binomial.

The conditional terms are Dirichlet-Multinomial.

Thus

\[
\mathcal L_{\text{Tree}}
=
\lambda_N\mathcal L_{NB}
+
\lambda_{64}\mathcal L_{N\rightarrow64}^{DM}
+
\lambda_{32}\mathcal L_{64\rightarrow32}^{DM}
+
\lambda_{16}\mathcal L_{32\rightarrow16}^{DM}.
\]

The proposed core loss is:

\[
\boxed{
\mathcal L_{\text{NTPC-core}}
=
\mathcal L_{\text{root}}
+
\mathcal L_{N\rightarrow64}^{DM}
+
\mathcal L_{64\rightarrow32}^{DM}
+
\mathcal L_{32\rightarrow16}^{DM}
}
\]

with default unit weights:

\[
w_{\text{root}}=
w_{64}=
w_{32}=
w_{16}=1.
\]

---

# 13. Root-to-64 allocation

The root has all valid \(64\times64\) regions as children.

Flatten

\[
Y^{64}
\rightarrow
\mathbf y^{64}
\]

and

\[
\mu^{64}
\rightarrow
\boldsymbol\mu^{64}.
\]

Then

\[
\pi^{64}_j
=
\frac{\max(\mu^{64}_j, \epsilon)}
{\sum_k \max(\mu^{64}_k, \epsilon)}.
\]

Use

\[
\mathbf Y^{64}\mid N
\sim
DM(
N,
\kappa_{64}\boldsymbol\pi^{64}
).
\]

This makes the root magnitude and root spatial composition distinct:

- Root NB: **how many people?**
- Root DM: **how is that total distributed over coarse regions?**

---

# 14. Optional density-adaptive fine likelihood (R5 Extension)

Fine \(8\times8\) supervision everywhere can be dominated by empty cells.

Therefore use the fixed base tree

\[
64\rightarrow32\rightarrow16
\]

as the primary proposed core formulation for all valid regions.

As an optional extension (R5), sufficiently dense \(16\times16\) parents receive additional

\[
16\rightarrow8
\]

likelihood.

Define the dense threshold from the **training split only**:

\[
\tau_D
=
Q_q
\left(
Y^{16}\mid Y^{16}>0
\right).
\]

Default:

\[
q=0.85.
\]

A parent is dense if

\[
p\in\mathcal D
\iff
Y_p^{16}\ge\tau_D.
\]

Then

\[
\mathcal L_{Dense8}
=
\frac1{|\mathcal D|}
\sum_{p\in\mathcal D}
-
\log
DM(
\mathbf Y^8_{child(p)}
\mid
Y_p^{16},
\kappa_8\pi_p
).
\]

The full adaptive extension loss (R5) is:

\[
\boxed{
\mathcal L_{\text{R5}}
=
\mathcal L_{\text{NTPC-core}}
+
\lambda_8
\mathcal L_{Dense8}
}
\]

with initial \(\lambda_8=1.0\).

The split is used **only to choose training likelihood terms**.

Inference remains pure single forward pass:

\[
\hat N=\sum D.
\]

---

# 15. Why this may help dense crowd counting

Suppose a \(16\times16\) parent contains

\[
Y_p=20
\]

people and its true four \(8\times8\) child counts are

\[
[12,5,2,1].
\]

Two predictions may have the same parent total:

\[
[5,5,5,5]
\]

and

\[
[11,6,2,1].
\]

A parent-only count loss sees both as perfectly correct:

\[
\sum_c\hat Y_c=20.
\]

The conditional allocation likelihood strongly prefers the second.

The intended mechanism is therefore:

> coarse count supervision fixes mass magnitude while conditional regional composition constrains *where that mass must be allocated*, especially inside congested areas where several plausible local arrangements can share the same total count.

The Dirichlet-tree model additionally allows different local splitting variability through \(\kappa_l\).

---

# 16. Flat Dirichlet-Multinomial baseline

The most important ablation against the tree is a flat DM over the finest base level.

Flatten all valid \(16\times16\) counts:

\[
\mathbf Y^{16}
\]

and predicted masses

\[
\boldsymbol\mu^{16}.
\]

Define

\[
\pi_j^{16}
=
\frac{\mu_j^{16}+\epsilon}
{\sum_k\mu_k^{16}+M\epsilon}.
\]

Then

\[
\mathbf Y^{16}\mid N
\sim
DM(N,\kappa\pi^{16}).
\]

This baseline has:

- exact counts;
- Gaussian-free supervision;
- probability;
- overdispersion;
- conservation;

but **no hierarchical conditional structure**.

Therefore:

\[
\boxed{
DTM\ must\ beat\ flat\ DM
}
\]

for the tree contribution to be convincing.

---

# 17. Scratch-training rule

All official experiments must follow:

```text
NO old checkpoint
NO resume from historical run
NO teacher model
NO knowledge distillation
NO inherited crowd-counting weights
NO checkpoint-preserving initialization trick
```

Strict scratch means:

```yaml
pretrained: false
checkpoint: null
resume: null
```

All compared formulations use the same initialization policy.

If a better initialization for the mass head is introduced, it must be applied to **all** baselines and proposed runs.

---

# 18. Mass-head initialization for scratch training

A positive mass map with Softplus can start with a dangerously large image count.

If logits start near zero,

\[
Softplus(0)=\log2\approx0.693.
\]

With \(M\) output cells,

\[
E[\hat N_0]\approx0.693M.
\]

For a \(64\times64\) mass map,

\[
\hat N_0\approx2839.
\]

This is undesirable.

Use a negative output bias.

A principled initialization is:

1. estimate average training crop count \(\bar N\);
2. let \(M\) be the number of stride-4 mass cells in a crop;
3. target initial mass per cell:

\[
d_0=\frac{\bar N}{M};
\]

4. initialize final bias to

\[
b_0
=
Softplus^{-1}(d_0)
=
\log(e^{d_0}-1).
\]

For numerical stability:

\[
Softplus^{-1}(x)
=
x+\log(-\operatorname{expm1}(-x)).
\]

This uses no learned weights and is fully compatible with training from scratch.

---

# 19. Official ablation matrix

The most informative experiment sequence is:

| Run | Formulation | Scientific question |
|---|---|---|
| R0 | Exact regional L1 | basic local-count supervision |
| R1 | Deterministic conserved allocation | is decomposition alone enough? |
| R2 | Flat DM at 16 | is probabilistic overdispersion enough? |
| R3 | Hierarchical Multinomial | Multinomial control: does DTM improve over the corresponding non-overdispersed allocation model? |
| R4 | Neural DTM 64→32→16 | does conditional over-dispersed hierarchy help? (Proposed Core) |
| R5 | R4 + dense-only 16→8 | does adaptive fine likelihood improve dense cases? (Optional Extension) |

Under a single conserved mass map, pure hierarchical Multinomial probabilities collapse to the flat leaf Multinomial. Therefore, R3 serves as a strict non-overdispersed allocation control to isolate the exact impact of Dirichlet concentration / overdispersion in DTM.

The two critical comparisons are

\[
R4\ vs\ R2
\]

and

\[
R5\ vs\ R4.
\]

If R4 does not beat R2 consistently, the tree hypothesis is weak.

If R5 does not improve dense-subset metrics, remove it.

Do not hide negative results.

---

# 20. Required reporting

For every official model report:

\[
MAE
=
\frac1M
\sum_i
|\hat N_i-N_i|
\]

\[
RMSE
=
\sqrt{
\frac1M
\sum_i
(\hat N_i-N_i)^2
}
\]

\[
Bias
=
\frac1M
\sum_i
(\hat N_i-N_i).
\]

Also report density-stratified metrics.

Recommended ShanghaiTech A groups:

```text
sparse:   N < 300
medium:   300 <= N < 1000
dense:    N >= 1000
```

The exact cutoffs may be changed, but they must be fixed before final comparison.

Report:

- overall MAE;
- overall RMSE;
- sparse MAE;
- medium MAE;
- dense MAE;
- dense RMSE;
- dense Bias;
- params;
- FLOPs at fixed resolution;
- latency on fixed hardware.

The first formulation/depth study uses one fixed seed:

```text
seed = 42
```

Do not launch multi-seed runs until R0--R5 and the localization depth study
have produced a complete first-pass table. Multi-seed verification is a later
confirmation phase for the surviving baseline and proposed model only.

---

## 20.1 Secondary parameter-free localization

Localization is an analysis capability of the conserved mass field, not a
learned branch of the core architecture:

\[
D \rightarrow \mathrm{OT\text{-}M} \rightarrow \{\hat p_i\}.
\]

Evaluate the same one-seed checkpoints trained with three hierarchy depths:

```text
R4 / DTM16 -> OT-M
T1 / DTM8  -> OT-M
T2 / DTM4  -> OT-M
```

For each checkpoint report a local-maximum baseline and OT-M with:

```text
Precision / Recall / F1 at sigma = 4, 8 pixels
distance-gated Hungarian one-to-one matching
micro aggregation over the evaluation split
|number of localized points - sum(D)|
model latency and OT-M post-processing latency separately
```

OT-M uses `m = floor(sum(D) + 0.5)`, epsilon-scaling `0.75`, blur `0.01`,
at most 16 alternating OT/M iterations, and density-weighted initialization.
Because Softplus makes every cell positive, large maps may use deterministic
mass-preserving grid aggregation before OT; the retained mass ratio and source
size must be logged. Localization hyperparameters are fixed before evaluation
and are never tuned on `test_data`.

The localization claim is retained only when deeper hierarchy improves F1
without unacceptable counting degradation. No localization head is added in
this first study.

For later key confirmation runs use at least 3 random seeds:

\[
mean\pm std.
\]

---

# 21. PyTorch reference implementation

The following code is intended as a clean reference implementation.

It should be integrated with the repository's existing dataset/model interfaces rather than blindly copied without shape checks.

---

## 21.1 Utility: stable inverse Softplus

```python
# ntpc/utils/math.py

import math
import torch


def inverse_softplus_scalar(x: float) -> float:
    """
    Numerically stable inverse of softplus for positive scalar x.
    softplus^{-1}(x) = x + log(1 - exp(-x)).
    """
    if x <= 0:
        raise ValueError("x must be > 0")
    return x + math.log(-math.expm1(-x))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    if torch.any(x <= 0):
        raise ValueError("all x must be > 0")
    return x + torch.log(-torch.expm1(-x))
```

---

## 21.2 Point rasterization

```python
# ntpc/data/point_targets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PointCountTargets:
    impulse: torch.Tensor
    counts: Dict[int, torch.Tensor]
    image_count: torch.Tensor
    padded_hw: Tuple[int, int]
    original_hw: Tuple[int, int]


def ceil_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def rasterize_points(
    points_xy: torch.Tensor,
    height: int,
    width: int,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    points_xy: [N, 2], coordinates (x, y) in original image pixels.
    Returns impulse map [1, H, W] whose sum equals number of valid points.
    """
    if device is None:
        device = points_xy.device

    out = torch.zeros((1, height, width), device=device, dtype=dtype)

    if points_xy.numel() == 0:
        return out

    x = torch.floor(points_xy[:, 0]).long()
    y = torch.floor(points_xy[:, 1]).long()

    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    x = x[valid]
    y = y[valid]

    flat_idx = y * width + x
    flat = out.view(-1)

    ones = torch.ones_like(flat_idx, dtype=dtype, device=device)
    flat.scatter_add_(0, flat_idx, ones)

    return out


def exact_block_counts(
    impulse: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    impulse: [1, H, W], H and W divisible by block_size.
    returns [1, H/B, W/B].
    """
    if impulse.ndim != 3 or impulse.shape[0] != 1:
        raise ValueError("impulse must have shape [1,H,W]")

    H, W = impulse.shape[-2:]

    if H % block_size != 0 or W % block_size != 0:
        raise ValueError("H,W must be divisible by block_size")

    x = impulse.unsqueeze(0)  # [1,1,H,W]

    counts = F.avg_pool2d(
        x,
        kernel_size=block_size,
        stride=block_size,
    ) * float(block_size * block_size)

    return counts.squeeze(0)


def build_point_count_targets(
    points_xy: torch.Tensor,
    height: int,
    width: int,
    levels: Iterable[int] = (8, 16, 32, 64),
    pad_multiple: int = 64,
    device=None,
) -> PointCountTargets:
    """
    Pads the target canvas on the bottom/right to a common multiple.
    The same padding convention must be applied to the image/model support.
    """
    Hp = ceil_to_multiple(height, pad_multiple)
    Wp = ceil_to_multiple(width, pad_multiple)

    impulse = rasterize_points(
        points_xy=points_xy,
        height=height,
        width=width,
        device=device,
    )

    impulse = F.pad(
        impulse,
        (0, Wp - width, 0, Hp - height),
        value=0.0,
    )

    counts = {}
    for B in levels:
        counts[B] = exact_block_counts(impulse, B)

    image_count = impulse.sum().reshape(1)

    return PointCountTargets(
        impulse=impulse,
        counts=counts,
        image_count=image_count,
        padded_hw=(Hp, Wp),
        original_hw=(height, width),
    )
```

---

## 21.3 Batched target construction

For fixed-size training crops, batch targets can be built directly.

```python
# ntpc/data/batch_targets.py

from typing import Dict, List

import torch

from .point_targets import build_point_count_targets


def build_batch_targets(
    batch_points: List[torch.Tensor],
    height: int,
    width: int,
    device,
) -> Dict:
    pyramids = {8: [], 16: [], 32: [], 64: []}
    totals = []

    for points in batch_points:
        t = build_point_count_targets(
            points_xy=points.to(device),
            height=height,
            width=width,
            levels=(8, 16, 32, 64),
            device=device,
        )

        for B in pyramids:
            pyramids[B].append(t.counts[B])

        totals.append(t.image_count)

    out = {
        f"y{B}": torch.stack(pyramids[B], dim=0)
        for B in pyramids
    }

    out["N"] = torch.stack(totals, dim=0).view(-1)

    return out
```

Expected shapes:

```text
y8  : [B,1,H/8,W/8]
y16 : [B,1,H/16,W/16]
y32 : [B,1,H/32,W/32]
y64 : [B,1,H/64,W/64]
N   : [B]
```

---

## 21.4 Predicted count pyramid

```python
# ntpc/losses/predicted_counts.py

from typing import Dict

import torch
import torch.nn.functional as F


def sum_pool_mass(
    mass: torch.Tensor,
    image_block_size: int,
    output_stride: int = 4,
) -> torch.Tensor:
    """
    mass: [B,1,H/4,W/4]
    """
    if image_block_size % output_stride != 0:
        raise ValueError("block size must be divisible by output stride")

    k = image_block_size // output_stride

    pooled = F.avg_pool2d(
        mass,
        kernel_size=k,
        stride=k,
    ) * float(k * k)

    return pooled


def predicted_count_pyramid(
    mass: torch.Tensor,
    levels=(8, 16, 32, 64),
    output_stride: int = 4,
) -> Dict[int, torch.Tensor]:
    return {
        B: sum_pool_mass(
            mass,
            image_block_size=B,
            output_stride=output_stride,
        )
        for B in levels
    }


def predicted_total_count(mass: torch.Tensor) -> torch.Tensor:
    return mass.flatten(1).sum(dim=1)
```

---

## 21.5 Negative Binomial NLL

```python
# hpc/losses/negative_binomial.py

from __future__ import annotations

import torch


_MAX_DISPERSION = 1e4


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """NB NLL with Var(Y)=mu+mu^2/r, evaluated in float32."""
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)
    r = torch.as_tensor(dispersion, device=mean.device, dtype=torch.float32)

    if torch.any(r <= 0) or torch.any(r > _MAX_DISPERSION) or not torch.isfinite(r).all():
        raise ValueError(
            f"Negative-Binomial dispersion parameter r must be in (0, {_MAX_DISPERSION}], got {dispersion}"
        )
    if torch.any(y < 0):
        raise ValueError("Negative-Binomial targets must be non-negative")

    log_r_plus_mu = torch.log(r + mu)
    nll = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def poisson_nll(y: torch.Tensor, mu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Elementwise Poisson NLL including the constant log(y!) term."""
    y = y.float()
    mu = mu.float().clamp_min(eps)
    if torch.any(y < 0):
        raise ValueError("Poisson targets must be non-negative")
    return mu - y * torch.log(mu) + torch.lgamma(y + 1.0)
```

---

## 21.6 Multinomial NLL

```python
# ntpc/losses/multinomial.py

import torch


def normalize_mass(
    child_mass: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    child_mass: [..., K]
    """
    K = child_mass.shape[-1]

    x = child_mass.float().clamp_min(0.0) + eps

    return x / x.sum(dim=-1, keepdim=True).clamp_min(K * eps)


def multinomial_nll(
    y: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    include_constant: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    y, pi: [M,K]
    """
    y = y.float()
    pi = pi.float().clamp_min(eps)

    n = y.sum(dim=-1)

    log_prob = (y * torch.log(pi)).sum(dim=-1)

    if include_constant:
        log_prob = (
            torch.lgamma(n + 1.0)
            - torch.lgamma(y + 1.0).sum(dim=-1)
            + log_prob
        )

    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll

    raise ValueError(reduction)
```

---

## 21.7 Dirichlet-Multinomial NLL

```python
# ntpc/losses/dirichlet_multinomial.py

import torch


def dirichlet_multinomial_nll(
    y: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    y:     [M,K] non-negative exact counts
    alpha: [M,K] strictly positive concentration parameters
    """
    y = y.float()
    alpha = alpha.float().clamp_min(eps)

    n = y.sum(dim=-1)
    alpha0 = alpha.sum(dim=-1)

    log_prob = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(y + 1.0).sum(dim=-1)
        + torch.lgamma(alpha0)
        - torch.lgamma(n + alpha0)
        + (
            torch.lgamma(y + alpha)
            - torch.lgamma(alpha)
        ).sum(dim=-1)
    )

    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll

    raise ValueError(reduction)


def dm_from_mass(
    y: torch.Tensor,
    child_mass: torch.Tensor,
    kappa: float,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Builds alpha = kappa * normalized(child_mass).
    """
    x = child_mass.float().clamp_min(0.0) + eps
    pi = x / x.sum(dim=-1, keepdim=True)
    alpha = float(kappa) * pi

    return dirichlet_multinomial_nll(
        y=y,
        alpha=alpha,
        reduction=reduction,
    )
```

Under AMP, keep the `lgamma` calculations in float32.

---

## 21.8 Group four children under every parent

```python
# ntpc/losses/tree_ops.py

import torch


def group_2x2_children(
    fine: torch.Tensor,
) -> torch.Tensor:
    """
    fine: [B,1,H,W]

    Returns:
        [B,H/2,W/2,4]

    child order:
        top-left, top-right, bottom-left, bottom-right
    """
    if fine.ndim != 4 or fine.shape[1] != 1:
        raise ValueError("expected [B,1,H,W]")

    B, _, H, W = fine.shape

    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("H and W must be even")

    x = fine[:, 0]

    tl = x[:, 0::2, 0::2]
    tr = x[:, 0::2, 1::2]
    bl = x[:, 1::2, 0::2]
    br = x[:, 1::2, 1::2]

    return torch.stack(
        [tl, tr, bl, br],
        dim=-1,
    )


def flatten_nonzero_parents(
    y_parent: torch.Tensor,
    y_children: torch.Tensor,
    pred_child_mass: torch.Tensor,
):
    """
    y_parent:        [B,1,H,W]
    y_children:      [B,H,W,4]
    pred_child_mass: [B,H,W,4]
    """
    parent = y_parent[:, 0]

    mask = parent > 0

    return (
        y_children[mask],
        pred_child_mass[mask],
        mask,
    )
```

---

## 21.9 Hierarchical DM term

```python
# ntpc/losses/tree_level.py

import torch

from .dirichlet_multinomial import dm_from_mass
from .tree_ops import group_2x2_children


def hierarchical_dm_level_loss(
    y_parent: torch.Tensor,
    y_child_map: torch.Tensor,
    pred_child_map: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """
    Example:
        y_parent     = y64
        y_child_map  = y32
        pred_child_map = mu32
    """
    y_child = group_2x2_children(y_child_map)
    pred_child = group_2x2_children(pred_child_map)

    parent = y_parent[:, 0]
    valid = parent > 0

    if valid.sum() == 0:
        return pred_child_map.sum() * 0.0

    y = y_child[valid]
    m = pred_child[valid]

    return dm_from_mass(
        y=y,
        child_mass=m,
        kappa=kappa,
        reduction="mean",
    )
```

---

## 21.10 Root-to-64 DM

```python
# ntpc/losses/root_allocation.py

import torch

from .dirichlet_multinomial import dm_from_mass


def root_to_64_dm_loss(
    y64: torch.Tensor,
    mu64: torch.Tensor,
    kappa64: float,
) -> torch.Tensor:
    """
    Each image uses one DM over all 64x64 blocks.

    y64, mu64: [B,1,H64,W64]
    """
    losses = []

    for b in range(y64.shape[0]):
        y = y64[b].reshape(1, -1)
        m = mu64[b].reshape(1, -1)

        n = y.sum()

        if n <= 0:
            continue

        losses.append(
            dm_from_mass(
                y=y,
                child_mass=m,
                kappa=kappa64,
                reduction="mean",
            )
        )

    if not losses:
        return mu64.sum() * 0.0

    return torch.stack(losses).mean()
```

For batches containing variable valid padded areas, flatten only valid \(64\times64\) cells.

---

## 21.11 Dense 16→8 DM loss

```python
# ntpc/losses/dense_fine.py

import torch

from .dirichlet_multinomial import dm_from_mass
from .tree_ops import group_2x2_children


def dense_16_to_8_dm_loss(
    y16: torch.Tensor,
    y8: torch.Tensor,
    mu8: torch.Tensor,
    dense_threshold: float,
    kappa8: float,
) -> torch.Tensor:
    """
    Dense decision is based ONLY on GT training counts.

    y16: [B,1,H16,W16]
    y8:  [B,1,H8,W8]
    mu8: [B,1,H8,W8]
    """
    y_child = group_2x2_children(y8)
    pred_child = group_2x2_children(mu8)

    parent = y16[:, 0]

    dense = parent >= float(dense_threshold)

    if dense.sum() == 0:
        return mu8.sum() * 0.0

    y = y_child[dense]
    m = pred_child[dense]

    return dm_from_mass(
        y=y,
        child_mass=m,
        kappa=kappa8,
        reduction="mean",
    )
```

---

## 21.12 Deterministic conserved allocation baseline

```python
# ntpc/losses/deterministic_allocation.py

import torch

from .multinomial import normalize_mass
from .tree_ops import group_2x2_children


def deterministic_allocation_level_loss(
    y_parent: torch.Tensor,
    y_child_map: torch.Tensor,
    pred_child_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Prediction is GT_parent * predicted_proportion.
    This isolates spatial allocation quality while conserving parent GT count.
    """
    y_child = group_2x2_children(y_child_map)
    pred_child_mass = group_2x2_children(pred_child_map)

    parent = y_parent[:, 0]
    valid = parent > 0

    if valid.sum() == 0:
        return pred_child_map.sum() * 0.0

    y = y_child[valid]
    m = pred_child_mass[valid]
    p = parent[valid].unsqueeze(-1)

    pi = normalize_mass(m, eps=eps)
    expected_child = p * pi

    local_l1 = torch.abs(
        expected_child - y
    ).sum(dim=-1)

    normalized = local_l1 / p.squeeze(-1).clamp_min(1.0)

    return normalized.mean()
```

---

## 21.13 Flat DM baseline at level 16

```python
# ntpc/losses/flat_dm.py

import torch

from .dirichlet_multinomial import dm_from_mass


def flat_dm_level_loss(
    y_level: torch.Tensor,
    mu_level: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """
    One DM per image across every regional cell at the selected level.
    """
    losses = []

    for b in range(y_level.shape[0]):
        y = y_level[b].reshape(1, -1)
        m = mu_level[b].reshape(1, -1)

        if y.sum() <= 0:
            continue

        losses.append(
            dm_from_mass(
                y=y,
                child_mass=m,
                kappa=kappa,
                reduction="mean",
            )
        )

    if not losses:
        return mu_level.sum() * 0.0

    return torch.stack(losses).mean()
```

---

## 21.14 Main NTPC loss

```python
# ntpc/losses/ntpc.py

from dataclasses import dataclass

import torch
import torch.nn as nn

from .negative_binomial import (
    negative_binomial_nll_mean_dispersion,
)
from .predicted_counts import (
    predicted_count_pyramid,
    predicted_total_count,
)
from .root_allocation import root_to_64_dm_loss
from .tree_level import hierarchical_dm_level_loss
from .dense_fine import dense_16_to_8_dm_loss


@dataclass
class NTPCConfig:
    root_dispersion: float = 50.0

    kappa64: float = 20.0
    kappa32: float = 20.0
    kappa16: float = 20.0
    kappa8: float = 20.0

    dense_threshold16: float = 4.0

    w_root: float = 1.0
    w_root64: float = 1.0
    w_64_32: float = 1.0
    w_32_16: float = 1.0
    w_dense8: float = 1.0

    enable_dense8: bool = True


class NTPCLoss(nn.Module):
    def __init__(self, cfg: NTPCConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        mass: torch.Tensor,
        targets: dict,
    ):
        """
        mass: [B,1,H/4,W/4]

        targets:
            N
            y8
            y16
            y32
            y64
        """
        cfg = self.cfg

        pred = predicted_count_pyramid(
            mass,
            levels=(8,16,32,64),
            output_stride=4,
        )

        muN = predicted_total_count(mass)

        L_root = negative_binomial_nll_mean_dispersion(
            y=targets["N"],
            mu=muN,
            r=cfg.root_dispersion,
        )

        L_root64 = root_to_64_dm_loss(
            y64=targets["y64"],
            mu64=pred[64],
            kappa64=cfg.kappa64,
        )

        L_64_32 = hierarchical_dm_level_loss(
            y_parent=targets["y64"],
            y_child_map=targets["y32"],
            pred_child_map=pred[32],
            kappa=cfg.kappa32,
        )

        L_32_16 = hierarchical_dm_level_loss(
            y_parent=targets["y32"],
            y_child_map=targets["y16"],
            pred_child_map=pred[16],
            kappa=cfg.kappa16,
        )

        if cfg.enable_dense8:
            L_dense8 = dense_16_to_8_dm_loss(
                y16=targets["y16"],
                y8=targets["y8"],
                mu8=pred[8],
                dense_threshold=cfg.dense_threshold16,
                kappa8=cfg.kappa8,
            )
        else:
            L_dense8 = mass.sum() * 0.0

        total = (
            cfg.w_root * L_root
            + cfg.w_root64 * L_root64
            + cfg.w_64_32 * L_64_32
            + cfg.w_32_16 * L_32_16
            + cfg.w_dense8 * L_dense8
        )

        logs = {
            "loss": total.detach(),
            "root_nb": L_root.detach(),
            "root_to_64_dm": L_root64.detach(),
            "64_to_32_dm": L_64_32.detach(),
            "32_to_16_dm": L_32_16.detach(),
            "dense_16_to_8_dm": L_dense8.detach(),
            "pred_count_mean": muN.detach().mean(),
            "gt_count_mean": targets["N"].detach().float().mean(),
        }

        return total, logs
```

---

# 22. Lightweight mass-head reference

The proposed learning formulation does not require a special head.

A minimal head:

```python
# ntpc/models/mass_head.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ntpc.utils.math import inverse_softplus_scalar


class MassHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        initial_mass_per_cell: float = 0.02,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.out = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        nn.init.normal_(
            self.out.weight,
            mean=0.0,
            std=1e-3,
        )

        bias = inverse_softplus_scalar(
            initial_mass_per_cell
        )

        nn.init.constant_(
            self.out.bias,
            bias,
        )

    def forward(self, x):
        z = self.out(self.body(x))
        return F.softplus(z)
```

The final `initial_mass_per_cell` must be computed from training crops if possible.

---

# 23. Model interface

Any backbone/neck may be used as long as it returns a stride-4 feature map.

```python
# ntpc/models/model.py

import torch.nn as nn

from .mass_head import MassHead


class NTPCModel(nn.Module):
    def __init__(
        self,
        backbone,
        neck,
        p4_channels: int,
        initial_mass_per_cell: float,
    ):
        super().__init__()

        self.backbone = backbone
        self.neck = neck

        self.mass_head = MassHead(
            in_channels=p4_channels,
            hidden_channels=32,
            initial_mass_per_cell=initial_mass_per_cell,
        )

    def forward(self, x):
        features = self.backbone(x)
        p4 = self.neck(features)
        mass = self.mass_head(p4)
        return mass
```

Inference:

```python
mass = model(image)
pred_count = mass.sum(dim=(1,2,3))
```

---

# 24. Training loop

```python
# train_ntpc.py

import torch
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(
    model,
    criterion,
    loader,
    optimizer,
    device,
    scaler: GradScaler,
):
    model.train()

    running = {}

    for batch in loader:
        image = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = {
            k: v.to(
                device,
                non_blocking=True,
            )
            for k, v in batch["targets"].items()
        }

        optimizer.zero_grad(
            set_to_none=True,
        )

        with autocast(
            enabled=True,
        ):
            mass = model(image)

            # lgamma-based loss code internally casts to float32.
            loss, logs = criterion(
                mass,
                targets,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        for k, v in logs.items():
            running.setdefault(k, []).append(
                float(v)
            )

    return {
        k: sum(v) / len(v)
        for k, v in running.items()
    }
```

---

# 25. Validation

```python
# ntpc/eval/count_metrics.py

import math

import torch


@torch.no_grad()
def evaluate_counting(
    model,
    loader,
    device,
):
    model.eval()

    errors = []
    sq_errors = []
    signed_errors = []

    for batch in loader:
        image = batch["image"].to(device)
        gt = batch["count"].to(device).float()

        mass = model(image)

        pred = mass.flatten(1).sum(dim=1)

        err = pred - gt

        errors.extend(
            err.abs().cpu().tolist()
        )

        sq_errors.extend(
            (err * err).cpu().tolist()
        )

        signed_errors.extend(
            err.cpu().tolist()
        )

    mae = sum(errors) / len(errors)
    rmse = math.sqrt(
        sum(sq_errors) / len(sq_errors)
    )
    bias = sum(signed_errors) / len(signed_errors)

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }
```

Validation must use **the exact same evaluation protocol for every run**.

---

# 26. Dense-stratified evaluation

```python
# ntpc/eval/stratified.py

import math


def metric_from_pairs(
    predictions,
    targets,
):
    err = [
        float(p) - float(t)
        for p, t in zip(
            predictions,
            targets,
        )
    ]

    mae = sum(abs(e) for e in err) / len(err)

    rmse = math.sqrt(
        sum(e * e for e in err) / len(err)
    )

    bias = sum(err) / len(err)

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def stratified_metrics(
    predictions,
    targets,
):
    groups = {
        "sparse": [],
        "medium": [],
        "dense": [],
    }

    for p, t in zip(
        predictions,
        targets,
    ):
        if t < 300:
            name = "sparse"
        elif t < 1000:
            name = "medium"
        else:
            name = "dense"

        groups[name].append((p, t))

    out = {}

    for name, pairs in groups.items():
        if not pairs:
            continue

        p, t = zip(*pairs)

        out[name] = metric_from_pairs(
            p,
            t,
        )

    return out
```

---

# 27. Computing the dense threshold

Use only training annotations.

```python
# scripts/estimate_dense_threshold.py

import torch


def estimate_dense_threshold(
    all_y16_positive_counts,
    quantile=0.85,
):
    x = torch.as_tensor(
        all_y16_positive_counts,
        dtype=torch.float32,
    )

    x = x[x > 0]

    if x.numel() == 0:
        raise RuntimeError(
            "No positive y16 cells found"
        )

    tau = torch.quantile(
        x,
        quantile,
    )

    return float(tau.item())
```

Recommended ablation:

```text
q = 0.75
q = 0.85
q = 0.90
```

---

# 28. Computing initial mass bias

```python
# scripts/estimate_mass_init.py

from ntpc.utils.math import inverse_softplus_scalar


def estimate_initial_mass_per_cell(
    mean_crop_count: float,
    crop_h: int,
    crop_w: int,
    output_stride: int = 4,
):
    cells = (
        (crop_h // output_stride)
        * (crop_w // output_stride)
    )

    mass_per_cell = (
        mean_crop_count / float(cells)
    )

    bias = inverse_softplus_scalar(
        mass_per_cell
    )

    return {
        "mass_per_cell": mass_per_cell,
        "final_bias": bias,
    }
```

This initialization must be derived from training statistics only.

---

# 29. Reference YAML configuration

Authoritative example from `configs/ntpc_r4_neural_dtm_tree.yaml`:

```yaml
experiment:
  name: ntpc_r4_neural_dtm_tree
  seed: 42
  save_dir: ./runs/ntpc_r4_neural_dtm_tree

dataset:
  name: sha
  part: part_A
  root: ./data/ShanghaiTech
  crop_size: 256
  coordinate_base: 1
  image_mean: [0.485, 0.456, 0.406]
  image_std: [0.229, 0.224, 0.225]

model:
  backbone: mobilenetv4_conv_small_050
  pretrained: false
  neck_width: 32
  context_dilations: [1, 2, 3]
  use_p8_context: false
  use_repblock: false
  eps_d: 1.0e-08
  output_stride: 4

statistics:
  seed: 12345
  root_dispersion: 50.0
  max_samples: null
  crops_per_image: 3

loss:
  mode: r4_dtm_tree16
  root_loss: nb
  kappa_shared: 20.0
  w_root_nb: 1.0
  w_root64: 1.0
  w_64_32: 1.0
  w_32_16: 1.0

augmentation:
  scale_range: [0.7, 1.3]
  flip_prob: 0.5

sampler:
  weighted: false

optimizer:
  name: AdamW
  lr: 0.0001
  weight_decay: 0.0001
  grad_clip: 5.0

schedule:
  epochs: 1000
  warmup_epochs: 25

training:
  batch_size: 16
  drop_last: true
  amp: true
  init_scale: 256.0
  num_workers: 0
  evaluate_every: 5
  gradient_audit_every: 50
```

Benchmark selection protocol used by every matched R0--R5 run:

```text
ShanghaiTech A: 300 train_data -> train; 182 test_data -> periodic evaluation/best MAE
ShanghaiTech B: 400 train_data -> train; 316 test_data -> periodic evaluation/best MAE
UCF-QNRF:       1201 Train -> train; 334 Test -> periodic evaluation/best MAE
NWPU/JHU:       use the official train/val/test files
UCF-CC50:       official 5-fold cross-validation
```

No custom split is carved out of the ShanghaiTech or UCF-QNRF training set. All
R0--R5 comparisons must use the same evaluation interval and checkpoint rule.

---

# 30. Unit tests

Before any long experiment, all tests below must pass.

---

## 30.1 Point conservation

```python
def test_point_count_conservation():
    import torch

    from ntpc.data.point_targets import (
        build_point_count_targets,
    )

    points = torch.tensor([
        [10.0, 10.0],
        [20.0, 30.0],
        [200.0, 100.0],
        [255.0, 255.0],
    ])

    t = build_point_count_targets(
        points,
        height=256,
        width=256,
    )

    assert t.impulse.sum().item() == 4

    for B in (8,16,32,64):
        assert t.counts[B].sum().item() == 4
```

---

## 30.2 Target hierarchy conservation

```python
def test_target_hierarchy_conservation():
    import torch

    from ntpc.data.point_targets import (
        build_point_count_targets,
    )
    from ntpc.losses.tree_ops import (
        group_2x2_children,
    )

    points = torch.rand(100, 2) * 256

    t = build_point_count_targets(
        points,
        height=256,
        width=256,
    )

    y16_from_8 = (
        group_2x2_children(
            t.counts[8].unsqueeze(0)
        )
        .sum(dim=-1)
        .unsqueeze(1)
    )

    assert torch.allclose(
        y16_from_8,
        t.counts[16].unsqueeze(0),
    )
```

---

## 30.3 Predicted mass conservation

```python
def test_predicted_mass_conservation():
    import torch

    from ntpc.losses.predicted_counts import (
        predicted_count_pyramid,
    )

    mass = torch.rand(
        2,
        1,
        64,
        64,
    )

    pred = predicted_count_pyramid(
        mass,
    )

    total = mass.flatten(1).sum(dim=1)

    for B in (8,16,32,64):
        pooled = pred[B].flatten(1).sum(dim=1)

        assert torch.allclose(
            pooled,
            total,
            atol=1e-5,
            rtol=1e-5,
        )
```

---

## 30.4 DM finite gradient

```python
def test_dm_gradient_is_finite():
    import torch

    from ntpc.losses.dirichlet_multinomial import (
        dm_from_mass,
    )

    mass = torch.tensor(
        [[1.0,2.0,3.0,4.0]],
        requires_grad=True,
    )

    y = torch.tensor(
        [[1.0,2.0,4.0,3.0]]
    )

    loss = dm_from_mass(
        y,
        mass,
        kappa=20.0,
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(
        mass.grad
    ).all()
```

---

## 30.5 Correct allocation should have lower DM loss

```python
def test_correct_allocation_lower_loss():
    import torch

    from ntpc.losses.dirichlet_multinomial import (
        dm_from_mass,
    )

    y = torch.tensor(
        [[12.0,5.0,2.0,1.0]]
    )

    good = torch.tensor(
        [[12.0,5.0,2.0,1.0]]
    )

    bad = torch.tensor(
        [[5.0,5.0,5.0,5.0]]
    )

    L_good = dm_from_mass(
        y,
        good,
        kappa=20.0,
    )

    L_bad = dm_from_mass(
        y,
        bad,
        kappa=20.0,
    )

    assert L_good < L_bad
```

---

# 31. Mandatory optimization audit before full training

Before a 1000-epoch experiment, run:

### Test O1 — initial count distribution

Measure at step 0:

```text
mean predicted count
median predicted count
p10 / p90
mean GT count
median GT count
```

A scratch model should not begin with predicted counts thousands above the target purely due to Softplus baseline mass.

### Test O2 — one-image overfit

Train one image until:

\[
|\hat N-N|
\]

becomes very small.

If this fails, do not start full experiments.

### Test O3 — ten-image overfit

The network should reduce training MAE dramatically on 10 images.

### Test O4 — gradient norm

Log separately:

```text
grad norm from root NB
grad norm from root->64
grad norm from 64->32
grad norm from 32->16
grad norm from dense8
```

No term should dominate by several orders of magnitude unnoticed.

### Test O5 — loss scale

Record raw loss magnitude for each component during first 1000 updates.

Do not tune lambda coefficients blindly.

---

# 32. Recommended experiment procedure

## Phase 0: implementation validation

```text
unit tests
→ one-image overfit
→ ten-image overfit
→ 20–50 epoch smoke run
```

## Phase 1: formulation study

```text
R0 exact regional regression
R1 deterministic conserved allocation
R2 flat DM
R3 hierarchical multinomial
R4 DTM
R5 DTM + dense8
```

Do not change architecture inside this phase.

## Phase 2: architecture search

Only after the best formulation is identified:

- improve lightweight backbone;
- improve neck;
- try structural reparameterization;
- try larger effective receptive field;
- improve local representation.

The formulation comparison must remain controlled.

## Phase 3: multi-seed verification

This phase is deferred until the one-seed formulation and localization tables
are complete. It is not part of the first experiment launch.

Run best baseline and proposed model on:

```text
seed 1
seed 2
seed 3
```

Report mean ± std.

---

# 33. Decision rules

Use explicit falsification rules.

### Keep DTM if

\[
MAE_{DTM}
<
MAE_{flatDM}
\]

consistently across seeds and the difference is larger than run-to-run noise.

### Keep dense 16→8 if

\[
MAE_{dense}
\]

improves clearly without unacceptable degradation in overall MAE/RMSE.

### Reject the tree claim if

flat DM performs equivalently or better.

### Reject dense-adaptive refinement if

it only shifts errors between sparse and dense subsets without improving the primary objective.

---

# 34. Suggested paper contribution wording

## Contribution 1

> We formulate point-supervised crowd counting as a hierarchical probabilistic count-allocation problem. A single non-negative neural mass map simultaneously parameterizes global crowd magnitude and the conditional composition of exact regional counts.

## Contribution 2

> We instantiate the formulation using a Negative-Binomial root and Dirichlet-Multinomial node-wise splitting, yielding an image-conditioned neural Dirichlet-tree likelihood that models over-dispersed regional crowd allocations while preserving count consistency by construction.

## Contribution 3

> We introduce density-adaptive likelihood depth, applying finer conditional allocation supervision only to congested regions, which avoids uniformly fine supervision dominated by empty cells without adding inference-time branches.

Only retain Contribution 3 if experiments support it.

---

# 35. Suggested method paragraph

A compact Method description:

> Given point annotations, we construct exact regional integer-count pyramids without Gaussian smoothing. The network predicts a single positive stride-4 mass map whose sums define all regional and global count estimates, making cross-scale count conservation exact by construction. We model the global image count with a Negative Binomial likelihood. Conditioned on each parent count, the distribution of its child counts is modeled using a Dirichlet-Multinomial whose mean composition is parameterized by the normalized predicted child masses. Applying these conditional likelihoods recursively forms a neural Dirichlet-tree count model. At the finest level, only high-density 16×16 parents receive 8×8 child-allocation supervision. The entire probabilistic hierarchy is used only for training; inference requires only summing the predicted mass map.

---

# 36. Comparison to key prior approaches

## DM-Count

DM-Count asks:

> how should predicted spatial mass be matched to point annotations without Gaussian targets?

NTPC asks:

> given a conserved predicted mass field, how should exact regional integer counts be probabilistically allocated across a spatial hierarchy?

Main distinction:

```text
DM-Count:
point measure ↔ predicted measure
via optimal transport

NTPC:
total count → coarse counts → fine counts
via conditional count distributions
```

## ZIP

ZIP models block counts using Zero-Inflated Poisson.

NTPC instead models:

\[
P(total)
\times
P(spatial\ composition\mid total)
\]

recursively.

ZIP is primarily block-wise count modeling.

NTPC is conditional count allocation.

## S-DCNet

S-DCNet decomposes high local counts and redistributes them spatially.

NTPC must therefore not claim decomposition itself as novel.

The distinction is:

```text
S-DCNet:
deterministic / learned spatial decomposition

NTPC:
conditional over-dispersed count-allocation likelihood
parameterized by a single conserved mass field
```

## PET

PET performs adaptive quadtree point querying during model computation.

NTPC dense refinement is a **training likelihood selection mechanism**.

No adaptive inference tree is used.

---

# 37. Lightweight benchmark target

A sub-0.5M result is not automatically novel.

Existing lightweight crowd counters already occupy this regime.

Therefore the paper should target the joint frontier:

\[
\boxed{
\text{low MAE}
+
\text{sub-0.5M params}
+
\text{point-only supervision}
+
\text{no Gaussian}
+
\text{scratch training}
}
\]

The most scientifically meaningful claim is not an arbitrary target such as MAE 50.

It is:

> under the same architecture and scratch protocol, the proposed probabilistic formulation consistently improves counting accuracy and dense-region error over strong deterministic and flat probabilistic baselines.

A strong absolute result then amplifies the paper.

---

# 38. Optional extensions — not part of core method

Only test after the core formulation is validated.

## 38.1 Hard-zero auxiliary

Potentially suppress false mass in GT-zero regions.

Do not include in the main method unless it provides a clear gain.

## 38.2 Local contrastive auxiliary

Could improve local density discrimination.

This has substantial prior art and should be framed only as an auxiliary optimization technique.

## 38.3 RepConv / large-kernel reparameterization

Potentially improves representation without increasing deploy-time branch count.

This belongs to architecture optimization, not the main methodological novelty.

## 38.4 Learned concentration

Eventually one may predict

\[
\kappa_{p}
\]

from image features.

Do **not** start here.

First prove that fixed \(\kappa_l\) DTM beats flat DM and deterministic allocation.

---

# 39. Future stronger extension: image-conditioned dispersion

If fixed DTM is successful, a second-generation model could predict node concentration:

\[
\kappa_p
=
Softplus(g_\phi(F_p))
+
\kappa_{min}.
\]

Then

\[
\alpha_{p,c}
=
\kappa_p\pi_{p,c}.
\]

Interpretation:

- high \(\kappa_p\): confident, low-dispersion allocation;
- low \(\kappa_p\): uncertain / heterogeneous allocation.

This would make the model predict both:

1. expected child proportions;
2. local allocation uncertainty.

However, this increases complexity and should only be attempted after the fixed-\(\kappa\) hypothesis is validated.

---

# 40. Research checklist

Before claiming success:

```text
[ ] all models trained from scratch
[ ] same architecture for R0-R5 formulation study
[ ] exact point-derived targets only
[ ] no Gaussian density target
[ ] target count conservation tests pass
[ ] prediction conservation tests pass
[ ] one-image overfit passes
[ ] ten-image overfit passes
[ ] initial Softplus mass audited
[ ] flat DM baseline implemented
[ ] deterministic decomposition baseline implemented
[ ] hierarchical Multinomial baseline implemented
[ ] DTM beats flat DM
[ ] dense8 improves dense metrics
[ ] complete one-seed (seed 42) R0-R5 table first
[ ] complete one-seed DTM16/DTM8/DTM4 localization table first
[ ] surviving key results later repeated with >=3 seeds
[ ] mean +/- std reported only in the later confirmation phase
[ ] params/FLOPs measured consistently
[ ] no validation/test statistics used to choose train thresholds
[ ] negative results retained
[ ] no unsupported "first" claim
```

---

# 41. Definition of success

The core scientific hypothesis is supported only if:

\[
\boxed{
\text{Hierarchical DTM}
>
\text{Flat DM}
>
\text{or deterministic allocation}
}
\]

under an otherwise matched experiment.

The strongest result would be:

\[
\boxed{
\text{sub-0.5M}
+
\text{scratch}
+
\text{point-only}
+
\text{Gaussian-free}
+
\text{competitive/sub-60 SHHA MAE}
}
\]

but the paper should be driven by the causal ablation, not by a pre-declared magic MAE number.

---

# 42. Minimal first implementation order

Implement in exactly this order:

```text
1. point target pyramid
2. predicted SumPool pyramid
3. conservation tests
4. root NB
5. flat DM
6. deterministic allocation
7. hierarchical Multinomial
8. hierarchical DTM
9. dense-only 16→8
10. one-seed OT-M localization depth study
11. later multi-seed confirmation of surviving methods
```

Do not add extra modules before step 8 produces a clear result.

---

# 43. Recommended first run

The first meaningful probabilistic run should be:

```yaml
method: flat_dm_16

root:
  nb: true

flat_dm:
  level: 16
  kappa: 20

dense8:
  enabled: false
```

Then:

```yaml
method: dtm

tree:
  root_to_64: true
  64_to_32: true
  32_to_16: true

kappa:
  64: 20
  32: 20
  16: 20

dense8:
  enabled: false
```

Only compare dense refinement after this comparison is understood.

---

# 44. Final proposed formulation

The final core equations are:

\[
D=f_\theta(I),\qquad D\ge0
\]

\[
\mu_R=\sum_{x\in R}D_x
\]

\[
N\sim NB(\mu_N,r),
\qquad
\mu_N=\sum D
\]

\[
\pi_{p,c}
=
\frac{\mu_{p,c}+\epsilon}
{\sum_j\mu_{p,j}+K\epsilon}
\]

\[
\alpha_{p,c}
=
\kappa_l\pi_{p,c}
\]

\[
\mathbf Y_{child(p)}
\mid
Y_p,I
\sim
DM(
Y_p,
\boldsymbol\alpha_p
)
\]

and

\[
\boxed{
\mathcal L
=
\mathcal L_{NB}
+
\mathcal L_{N\rightarrow64}^{DM}
+
\mathcal L_{64\rightarrow32}^{DM}
+
\mathcal L_{32\rightarrow16}^{DM}
+
\lambda_8
\mathcal L_{16\rightarrow8,\ dense}^{DM}
}
\]

with

\[
\hat N=\sum D
\]

at inference.

---

# 45. One-sentence paper identity

> **NTPC trains an ultra-lightweight crowd counter by interpreting a single conserved neural mass map as the parameterization of an image-conditioned hierarchical count-splitting process over exact point-derived regional counts.**

---

# 46. Key references to verify during paper writing

These links are starting points, not a substitute for a final systematic related-work audit.

- Distribution Matching for Crowd Counting (DM-Count):  
  https://arxiv.org/abs/2009.13077

- Spatial Divide-and-Conquer / SS-DCNet:  
  https://arxiv.org/abs/2001.01886

- Point-Query Quadtree (PET):  
  https://arxiv.org/abs/2308.13814

- ZIP: Scalable Crowd Counting via Zero-Inflated Poisson Modeling:  
  https://arxiv.org/abs/2506.19955

- Dirichlet-Tree Multinomial Regression:  
  https://doi.org/10.1111/biom.12654

- Tree Pólya Splitting distributions for multivariate count data:  
  https://arxiv.org/abs/2404.19528

- Local Information Matters (LIMM):  
  https://arxiv.org/abs/2508.16970

---

# 47. Final warning

Do not let the paper become:

```text
tiny backbone
+ FPN
+ DTM
+ hard-zero
+ contrastive
+ attention
+ RepConv
+ dynamic routing
+ many auxiliary losses
```

That weakens the contribution.

The core scientific story should remain:

```text
Crowd count is spatially decomposable
        ↓
independent local regression ignores conditional composition
        ↓
model global magnitude once
        ↓
model regional allocation conditionally
        ↓
use a Dirichlet-tree likelihood
        ↓
deepen the likelihood only where crowd density warrants it
```

This gives a single falsifiable hypothesis, a clear mathematical mechanism, and a controlled path to an A*-quality experimental story if the results are strong enough.
