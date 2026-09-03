# MICF: Measure-Consistent Integral Count Fields for Ultra-Lightweight Crowd Counting

## 0. Status

**Research status:** Candidate A

**Core hypothesis:** predicting a cumulative spatial count field may be easier for severely capacity-limited crowd counters than directly predicting sparse/high-frequency local count targets, despite introducing stronger non-local dependency.

This document defines:
- exact mathematical formulation;
- scientific controls;
- MICF-v1 and MICF-v2 architectures;
- losses and constraints;
- PyTorch implementation skeletons;
- training/evaluation protocol;
- ablations and kill rules;
- defensible paper claims.

---

# 1. Research Questions

Let the exact local count map be

\[
Y\in\mathbb{N}_0^{H\times W},
\]

constructed directly from point annotations without Gaussian smoothing.

Conventional local-count learning predicts

\[
I\rightarrow \hat Y.
\]

MICF instead predicts

\[
\boxed{
C_{ij}
=
\sum_{a\le i}\sum_{b\le j}Y_{ab}
}
\]

and learns

\[
\boxed{I\rightarrow \hat C.}
\]

Main questions:

### RQ1 — Representation
Does directly predicting \(C\) outperform directly predicting \(Y\) under a matched tiny architecture?

### RQ2 — Loss vs. representation
Is any gain caused only by cumulative supervision

\[
L(P\hat Y,PY)
\]

or by predicting \(\hat C\) itself?

### RQ3 — Capacity interaction
Does the benefit increase as model capacity decreases?

### RQ4 — Context trade-off
When does reduced target complexity outweigh the non-local receptive-field requirement induced by \(C\)?

---

# 2. Exact Construction from Point Annotations

Suppose image \(I\) contains point annotations

\[
\mathcal P=\{(x_n,y_n)\}_{n=1}^{N}.
\]

Choose output stride \(s\).

For output grid \(H_o\times W_o\),

\[
Y_{ij}
=
\#\left\{
n:
\left\lfloor \frac{y_n}{s}\right\rfloor=i,\;
\left\lfloor \frac{x_n}{s}\right\rfloor=j
\right\}.
\]

Then

\[
\boxed{\sum_{ij}Y_{ij}=N.}
\]

No Gaussian density is required.

The cumulative target is

\[
\boxed{
C=\operatorname{CumSum}_y(
\operatorname{CumSum}_x(Y)
).
}
\]

PyTorch:

```python
C = Y.cumsum(dim=-2).cumsum(dim=-1)
```

---

# 3. Rectangle Count Recovery

For rectangle

\[
R=(x_1,x_2]\times(y_1,y_2],
\]

the exact rectangle count is

\[
\boxed{
N(R)
=
C(y_2,x_2)
-
C(y_1,x_2)
-
C(y_2,x_1)
+
C(y_1,x_1)
}
\]

with zero-padded boundaries.

Thus a correct cumulative field represents counts for all axis-aligned rectangles.

---

# 4. Linear Operator View

Let

\[
y=\operatorname{vec}(Y).
\]

Let \(T_H,T_W\) be lower-triangular cumulative-sum matrices. Then

\[
\boxed{
C=T_HYT_W^\top.
}
\]

Vectorized:

\[
\boxed{
c=Py,\qquad
P=T_W\otimes T_H.
}
\]

This transform is invertible.

If \(D=T^{-1}\) is the first-difference matrix,

\[
\boxed{
Y=D_HCD_W^\top.
}
\]

Equivalently,

\[
\boxed{
Y=\Delta_{xy}C.
}
\]

MICF therefore adds no information. Its possible contribution is a change in:
- target geometry;
- optimization geometry;
- spatial dependency structure;
- inductive bias under limited capacity.

---

# 5. Mixed Difference and Measure Consistency

Define

\[
\Delta_{xy}C_{ij}
=
C_{ij}
-
C_{i-1,j}
-
C_{i,j-1}
+
C_{i-1,j-1}.
\]

For a valid cumulative counting measure,

\[
\boxed{
\Delta_{xy}C_{ij}\ge0.
}
\]

With boundaries

\[
C_{0,j}=0,\qquad C_{i,0}=0,
\]

this is sufficient to imply coordinatewise monotonicity.

Hence separate x/y monotonicity penalties are unnecessary.

Reconstructed local mass:

\[
\boxed{
\hat Y=\Delta_{xy}\hat C.
}
\]

---

# 6. The Triangle Kill-Test

## A. Local output + local loss

\[
I\rightarrow\hat Y
\]

with

\[
\boxed{
L_A=L(\hat Y,Y).
}
\]

## B. Local output + cumulative loss

\[
I\rightarrow\hat Y
\]

but supervise

\[
\boxed{
L_B=L(P\hat Y,PY).
}
\]

This isolates integral-domain loss geometry.

## C. Direct MICF

\[
I\rightarrow\hat C
\]

with

\[
\boxed{
L_C=L(\hat C,PY).
}
\]

This isolates the output representation.

Interpretation:

\[
C>B>A
\]

→ representation hypothesis survives.

\[
B\approx C>A
\]

→ direct cumulative output is unnecessary; cumulative loss is the useful component.

\[
B>C
\]

→ integral supervision may help, but direct cumulative prediction is hurt by non-local dependency.

\[
A\ge B,C
\]

→ kill the integral-domain hypothesis.

---

# 7. Cumulative Loss as a Non-Local Error Metric

Let

\[
e=\hat y-y.
\]

Local squared loss:

\[
L_{\mathrm{local}}=e^\top e.
\]

Cumulative squared loss:

\[
\boxed{
L_{\mathrm{cum}}
=
\|Pe\|_2^2
=
e^\top P^\top P e.
}
\]

Thus cumulative supervision induces a correlated non-local error metric rather than independent cell-wise regression.

---

# 8. Origin Bias

For top-left cumulative supervision, cells near the top-left appear in more prefixes.

In 1D:

\[
(T^\top T)_{ik}
=
n-\max(i,k)+1.
\]

In 2D:

\[
\boxed{
K((i,j),(k,l))
=
(H-\max(i,k)+1)
(W-\max(j,l)+1).
}
\]

Therefore single-origin cumulative supervision introduces positional weighting.

---

# 9. Four-Orientation Balancing

Use TL/TR/BL/BR cumulative orientations.

For a point at \((x,y)\), total contribution across all four orientations is

\[
(H-x)(W-y)+(H-x)y+x(W-y)+xy=HW.
\]

A cheap implementation:

1. random horizontal flip;
2. random vertical flip;
3. transform point annotations;
4. regenerate the top-left cumulative target after augmentation.

No four-head architecture is required.

---

# 10. Dynamic Range

Raw cumulative targets satisfy

\[
0\le C_{ij}\le N.
\]

Avoid primary-target transforms such as

\[
\frac{C_{ij}}{(i+1)(j+1)}
\]

or

\[
\log(1+C),
\]

because they destroy exact linear rectangle recovery.

Preferred starting loss:

\[
\boxed{
L_{\mathrm{field}}
=
\operatorname{SmoothL1}(\hat C,C).
}
\]

If optimization needs scaling, use one scalar shared by the entire crop/image, not a position-dependent transform.

---

# 11. MICF-v1

Minimal scientific control:

\[
I
\rightarrow
F
\rightarrow
1\times1
\rightarrow
\hat C.
\]

No special context, attention, MoE, tree, or Gaussian target.

---

# 12. MICF-v2: Representation-Aligned Context

A location \((i,j)\) needs information about its prefix

\[
[0,i]\times[0,j].
\]

Generic GAP is insufficient because it loses prefix-specific spatial structure.

Define directional integral features:

\[
F^{TL}_{ij}
=
\sum_{a\le i,b\le j}F_{ab}.
\]

Normalize **features** by prefix area:

\[
\bar F^{TL}_{ij}
=
\frac{F^{TL}_{ij}}{(i+1)(j+1)}.
\]

Similarly form TR, BL, BR.

Fuse:

\[
\boxed{
F'
=
\phi(
F,
\bar F^{TL},
\bar F^{TR},
\bar F^{BL},
\bar F^{BR}
).
}
\]

Then

\[
\hat C=h(F').
\]

This context module is optional and must be ablated against generic context.

---

# 13. PyTorch Utilities

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Exact local count map

```python
def points_to_count_map(
    points_xy,
    out_h,
    out_w,
    stride,
    device=None,
    dtype=torch.float32,
):
    y = torch.zeros((out_h, out_w), device=device, dtype=dtype)

    if points_xy is None or len(points_xy) == 0:
        return y

    pts = torch.as_tensor(points_xy, device=device, dtype=torch.float32)

    gx = torch.floor(pts[:, 0] / stride).long()
    gy = torch.floor(pts[:, 1] / stride).long()

    valid = (
        (gx >= 0) & (gx < out_w) &
        (gy >= 0) & (gy < out_h)
    )

    gx = gx[valid]
    gy = gy[valid]

    if gx.numel() == 0:
        return y

    flat_idx = gy * out_w + gx
    y.view(-1).scatter_add_(
        0,
        flat_idx,
        torch.ones_like(flat_idx, dtype=dtype)
    )

    return y
```

## 2D cumulative sum

```python
def integral_2d(x):
    return x.cumsum(dim=-2).cumsum(dim=-1)
```

## Mixed finite difference

```python
def mixed_difference_2d(c):
    c_pad = F.pad(c, (1, 0, 1, 0))

    br = c_pad[..., 1:, 1:]
    bl = c_pad[..., 1:, :-1]
    tr = c_pad[..., :-1, 1:]
    tl = c_pad[..., :-1, :-1]

    return br - bl - tr + tl
```

Exact test:

```python
y = torch.rand(2, 1, 16, 16)
c = integral_2d(y)
y_rec = mixed_difference_2d(c)

assert torch.allclose(y, y_rec, atol=1e-6)
```

---

# 14. Orientation Utilities

```python
def integral_tl(x):
    return x.cumsum(-2).cumsum(-1)

def integral_tr(x):
    xr = torch.flip(x, dims=[-1])
    cr = xr.cumsum(-2).cumsum(-1)
    return torch.flip(cr, dims=[-1])

def integral_bl(x):
    xb = torch.flip(x, dims=[-2])
    cb = xb.cumsum(-2).cumsum(-1)
    return torch.flip(cb, dims=[-2])

def integral_br(x):
    xbr = torch.flip(x, dims=[-2, -1])
    cbr = xbr.cumsum(-2).cumsum(-1)
    return torch.flip(cbr, dims=[-2, -1])
```

---

# 15. Prefix-Normalized Feature Context

```python
def prefix_area(h, w, device, dtype):
    yy = torch.arange(1, h + 1, device=device, dtype=dtype)
    xx = torch.arange(1, w + 1, device=device, dtype=dtype)
    return yy[:, None] * xx[None, :]


def normalized_integral_tl(x, eps=1e-6):
    h, w = x.shape[-2:]
    area = prefix_area(h, w, x.device, x.dtype)
    return integral_tl(x) / (area + eps)


def normalized_integral_tr(x, eps=1e-6):
    xr = torch.flip(x, [-1])
    y = normalized_integral_tl(xr, eps)
    return torch.flip(y, [-1])


def normalized_integral_bl(x, eps=1e-6):
    xb = torch.flip(x, [-2])
    y = normalized_integral_tl(xb, eps)
    return torch.flip(y, [-2])


def normalized_integral_br(x, eps=1e-6):
    xbr = torch.flip(x, [-2, -1])
    y = normalized_integral_tl(xbr, eps)
    return torch.flip(y, [-2, -1])
```

---

# 16. Integral Context Block

```python
class IntegralContextBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = channels

        in_channels = channels * 5

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.dw = nn.Sequential(
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

        self.project = nn.Conv2d(hidden_channels, channels, 1, bias=False)

    def forward(self, x):
        tl = normalized_integral_tl(x)
        tr = normalized_integral_tr(x)
        bl = normalized_integral_bl(x)
        br = normalized_integral_br(x)

        z = torch.cat([x, tl, tr, bl, br], dim=1)
        z = self.reduce(z)
        z = self.dw(z)
        z = self.project(z)

        return x + z
```

This operates on features, not the target.

---

# 17. Minimal Heads

## Local count head

```python
class LocalCountHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return F.softplus(self.head(x))
```

## Direct MICF head

```python
class DirectMICFHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return self.head(x)
```

Do not confuse elementwise positivity with measure validity.

---

# 18. MICF-v2 Head

```python
class MICFv2Head(nn.Module):
    def __init__(self, in_channels, use_context=True):
        super().__init__()

        self.context = (
            IntegralContextBlock(in_channels)
            if use_context
            else nn.Identity()
        )

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        x = self.context(x)
        x = self.pre(x)
        return self.out(x)
```

---

# 19. Losses

## Local count loss

```python
def local_count_loss(pred_y, target_y):
    return F.smooth_l1_loss(pred_y, target_y)
```

## Local output + cumulative supervision

```python
def cumulative_supervision_loss(pred_y, target_y):
    pred_c = integral_2d(pred_y)
    target_c = integral_2d(target_y)
    return F.smooth_l1_loss(pred_c, target_c)
```

## Direct MICF field loss

```python
def micf_field_loss(pred_c, target_y):
    target_c = integral_2d(target_y)
    return F.smooth_l1_loss(pred_c, target_c)
```

## Measure-validity loss

```python
def measure_validity_loss(pred_c):
    pred_y = mixed_difference_2d(pred_c)
    return F.relu(-pred_y).mean()
```

## Optional local reconstruction loss

```python
def reconstructed_local_loss(pred_c, target_y):
    pred_y = mixed_difference_2d(pred_c)
    return F.smooth_l1_loss(pred_y, target_y)
```

Use the reconstruction term only as a later ablation.

---

# 20. Recommended Loss Schedule

## MICF-v1

\[
\boxed{
L=L_{\mathrm{field}}.
}
\]

## MICF + validity

\[
\boxed{
L=
L_{\mathrm{field}}
+
\lambda_v L_{\mathrm{valid}}.
}
\]

Initial sweep:

\[
\lambda_v\in\{0.01,0.05,0.1,0.5\}.
\]

## MICF-v2

\[
\boxed{
L=
L_{\mathrm{field}}
+
\lambda_vL_{\mathrm{valid}}
+
\lambda_yL_{\mathrm{local-recon}}.
}
\]

Suggested:

\[
\lambda_y\in\{0,0.01,0.05\}.
\]

---

# 21. Count Readout

For a TL cumulative field:

\[
\boxed{
\hat N_{corner}=\hat C_{H,W}.
}
\]

Also reconstruct

\[
\hat Y=\Delta_{xy}\hat C
\]

and compute

\[
\boxed{
\hat N_{\Delta}
=
\sum_{ij}\hat Y_{ij}.
}
\]

Diagnostic inconsistency:

\[
E_{cons}
=
|\hat N_{corner}-\hat N_{\Delta}|.
\]

---

# 22. Measure Diagnostics

Negative mass ratio:

\[
r_-=
\frac{
\sum[-\hat Y]_+
}{
\sum|\hat Y|+\epsilon
}.
\]

Negative-cell fraction:

\[
f_-=
\frac{\#\{\hat Y_{ij}<0\}}{HW}.
\]

Violation magnitude:

\[
V=
\frac1{HW}
\sum[-\hat Y_{ij}]_+.
\]

---

# 23. Rectangle Evaluation

Evaluate rectangle count error directly from \(\hat C\) at multiple scales.

Recommended normalized rectangle area bins:

\[
\left\{
\frac1{64},
\frac1{16},
\frac14,
1
\right\}.
\]

This tests whether MICF preserves region-level counting.

---

# 24. Capacity Sweep

Suggested approximate model sizes:

\[
\boxed{
0.05M,\;
0.10M,\;
0.25M,\;
0.8M
}
\]

with fixed architecture family and width scaling only.

Expected signature:

\[
\boxed{
\text{MICF gain largest at the smallest capacity.}
}
\]

If MICF only helps large models, the ultra-lightweight thesis weakens.

---

# 25. Receptive-Field Sweep

Separate RF from parameter count.

Examples:

- dilations \(\{1,1,1\}\);
- \(\{1,2,3\}\);
- \(\{1,3,5\}\);
- optional matched large-kernel depthwise convolution.

Question:

\[
\boxed{
\text{How much context is needed before cumulative prediction becomes useful?}
}
\]

---

# 26. Suggested Backbone

For formulation study:

```text
MobileNetV4 Small 0.5
    ↓
additive FPN
    ↓
32 channels
    ↓
DW dilation blocks {1,2,3}
    ↓
output stride 16 / 8 / 4
```

Keep the core free from:
- attention;
- MoE;
- dynamic routing;
- RepConv;
- distillation;
- pretrained weights.

---

# 27. Training Steps

## A — local

```python
pred_y = model(images)
loss = local_count_loss(pred_y, target_y)
```

## B — local output + cumulative loss

```python
pred_y = model(images)
loss = cumulative_supervision_loss(pred_y, target_y)
```

## C — direct MICF

```python
pred_c = model(images)

loss_field = micf_field_loss(pred_c, target_y)
loss_valid = measure_validity_loss(pred_c)

loss = loss_field + lambda_valid * loss_valid
```

For strict triangle control, start C with:

```python
lambda_valid = 0.0
```

---

# 28. Orientation-Balanced Training

```python
if torch.rand(()) < 0.5:
    image = torch.flip(image, dims=[-1])
    points[:, 0] = image_width - 1 - points[:, 0]

if torch.rand(()) < 0.5:
    image = torch.flip(image, dims=[-2])
    points[:, 1] = image_height - 1 - points[:, 1]
```

Then regenerate \(Y\) and \(C\).

---

# 29. Full-Image Inference Problem

For bottom-right:

\[
C(H,W)=N.
\]

A tiny local CNN cannot generally predict this if its effective receptive field is much smaller than the image.

Distinguish:

### Regime A — fixed-crop evaluation
Used to isolate representation geometry.

### Regime B — full-image evaluation
Requires a globally consistent spatial composition mechanism.

Generic GAP alone is not enough because each \(C(i,j)\) requires prefix-specific global information.

---

# 30. Hierarchical Tile Composition Idea

Future extension:

1. divide image into tiles;
2. predict local cumulative field \(C_t^{local}\);
3. obtain each tile total \(N_t\);
4. build a cumulative field over tile totals;
5. combine completed-tile mass with current-tile local field.

Conceptually:

\[
\boxed{
C_{global}
=
C_{completed\ tiles}
+
C_{current\ tile}.
}
\]

This may reduce the global RF burden, but it must not contaminate the initial triangle experiment.

---

# 31. Alternative Context: Axial Integral Context

Cheaper than full 2D feature-prefix context:

\[
R_{ij}
=
\frac1{j+1}
\sum_{b\le j}F_{ib},
\]

\[
V_{ij}
=
\frac1{i+1}
\sum_{a\le i}F_{aj}.
\]

Fuse:

\[
F'=\phi(F,R,V).
\]

Ablate axial vs. full 2D integral context.

---

# 32. Valid-by-Construction Control

Predict a local positive map:

\[
M=\operatorname{softplus}(h(F))
\]

then

\[
\boxed{
C=\operatorname{Integral}(M).
}
\]

This guarantees

\[
\Delta_{xy}C=M\ge0.
\]

But this is a **baseline**, not the main MICF method.

If it matches direct MICF, then direct cumulative representation has little independent value.

---

# 33. Required Baselines

Minimum:

- B0: Gaussian density regression;
- B1: exact local count regression;
- B2: exact local output + cumulative supervision;
- B3: direct MICF;
- B4: direct MICF + validity;
- B5: direct MICF + orientation balancing;
- B6: local positive map → deterministic integral;
- B7: matched ZIP.

Optional:
- DM-Count;
- PML;
- frequency-domain loss;
- MESA-style region supervision.

---

# 34. Ablation Matrix

| ID | Output | Cum loss | Validity | Orientation | Integral context |
|---|---|---:|---:|---:|---:|
| A | \(Y\) | no | n/a | same aug | no |
| B | \(Y\) | yes | n/a | same aug | no |
| C | \(C\) | yes | no | no | no |
| D | \(C\) | yes | yes | no | no |
| E | \(C\) | yes | yes | yes | no |
| F | \(C\) | yes | yes | yes | TL context |
| G | \(C\) | yes | yes | yes | 4-dir context |

---

# 35. Metrics

Primary:

\[
MAE
=
\frac1M\sum_i|\hat N_i-N_i|.
\]

\[
RMSE
=
\sqrt{
\frac1M
\sum_i(\hat N_i-N_i)^2
}.
\]

Optional NAE:

\[
NAE
=
\frac1M
\sum_i
\frac{|\hat N_i-N_i|}{N_i+\epsilon}.
\]

MICF-specific:
- prefix MAE;
- reconstructed local MAE;
- rectangle MAE by scale;
- negative-cell fraction;
- negative-mass ratio;
- corner-vs-difference count gap;
- positional error;
- density-bin error;
- representation × capacity interaction.

---

# 36. Spectral Analysis

Treat spectral analysis as supporting evidence, not the core theorem.

Compare:
- \(Y\);
- \(C\);
- local prediction errors;
- MICF errors.

Possible high-frequency energy:

\[
E_{high}
=
\frac{
\sum_{\|\omega\|>\tau}|\hat f(\omega)|^2
}{
\sum_\omega|\hat f(\omega)|^2
}.
\]

Also measure coefficient fraction required to retain 90%, 95%, 99% energy.

---

# 37. Dataset Progression

Stage 1:
- ShanghaiTech Part A.

Stage 2:
- UCF-QNRF.

Stage 3:
- NWPU-Crowd.

Optional sparse control:
- ShanghaiTech Part B.

---

# 38. Validation Protocol

Avoid test-set model selection.

For SHA:

1. split official training set into train/validation;
2. select hyperparameters on validation;
3. lock configuration;
4. retrain on full training set if desired;
5. evaluate test once.

Use at least

\[
\boxed{3\text{ seeds}.}
\]

Report mean ± standard deviation.

---

# 39. Optimization

Initial controlled schedule:

```text
optimizer: AdamW
lr: 1e-4
warmup: 25 epochs
epochs: up to 1000
gradient clipping: 5.0
```

LR sweep:

\[
\{10^{-4},3\times10^{-4},10^{-3}\}.
\]

Log:
- gradient norm;
- clip frequency;
- train loss;
- validation MAE;
- field dynamic range;
- invalid mass fraction.

---

# 40. Minimal Triangle Pilot

Dataset:
- ShanghaiTech A.

Crop:
- \(256\times256\).

Output:
- \(16\times16\) at stride 16.

Carrier:
- same ~0.1M backbone.

Variants:

### A
```text
output = Y
loss = SmoothL1(Y_hat, Y)
```

### B
```text
output = Y
loss = SmoothL1(Integral(Y_hat), Integral(Y))
```

### C
```text
output = C
loss = SmoothL1(C_hat, Integral(Y))
```

No special context, validity, or rescue mechanism in the strict control.

---

# 41. Stage-2 Development

If C is competitive with or better than B:

1. add validity;
2. add orientation balancing;
3. add integral context;
4. sweep RF;
5. sweep capacity;
6. add QNRF;
7. add ZIP;
8. add deterministic-integral control.

---

# 42. Kill Rules

Kill direct MICF representation if

\[
B\ge C
\]

consistently across datasets/capacities.

Kill integral-domain thesis if

\[
A\ge B,C.
\]

Kill ultra-lightweight claim if MICF only becomes useful for larger carriers.

Kill validity contribution if it only reduces violations without improving counting or regional consistency.

Kill integral-context novelty if matched generic context performs equally well.

---

# 43. GO Rules

Strong evidence requires:

1. direct MICF beats local-count baseline;
2. direct MICF beats local-output + cumulative-loss control;
3. gains repeat across at least two datasets;
4. gains are stable across at least three seeds;
5. gains are strongest in the smallest models;
6. validity improves measure consistency;
7. orientation balancing reduces positional bias;
8. integral context helps specifically under limited RF.

The strongest result is an interaction:

\[
\boxed{
\text{Representation}\times\text{Capacity}.
}
\]

---

# 44. Claim Boundaries

Safe:
- direct spatial cumulative count-field prediction for capacity-constrained crowd counting;
- invertible target reparameterization;
- loss-vs-representation decomposition;
- non-local dependency trade-off;
- mixed-difference measure consistency;
- orientation bias analysis.

Unsafe without evidence:
- first neural cumulative representation;
- first integral-count method in crowd counting;
- cumulative fields contain more information;
- integration guarantees easier optimization;
- universally better representation;
- SOTA localization.

---

# 45. Possible Contributions

If results support the hypothesis:

1. spatial cumulative count-field formulation;
2. controlled decomposition of loss geometry vs. output representation;
3. mixed-difference measure consistency;
4. orientation balancing;
5. capacity/RF interaction study;
6. optional lightweight integral-context block.

---

# 46. Proposed Titles

**Counting in the Integral Domain: When Does Cumulative Supervision Help Capacity-Limited Crowd Counters?**

**MICF: Measure-Consistent Integral Count Fields for Lightweight Crowd Counting**

**Integrate the Target, Not the Model: Cumulative Count Fields for Ultra-Lightweight Crowd Counting**

---

# 47. Expected Failure Modes

### A. Insufficient receptive field
- BR field errors are high;
- larger RF strongly helps MICF.

### B. Representation adds nothing
\[
B\approx C.
\]

### C. Invalid measure
Large
\[
\Delta_{xy}\hat C<0.
\]

### D. Orientation bias
Error depends strongly on cumulative origin.

### E. Better counting, worse localization
Global MAE improves while \(\Delta_{xy}\hat C\) is noisy.

This is scientifically meaningful rather than automatically a failure.

---

# 48. Unit Tests

```python
def test_integral_inverse():
    y = torch.randint(
        0,
        5,
        (4, 1, 16, 16),
        dtype=torch.float32,
    )

    c = integral_2d(y)
    y2 = mixed_difference_2d(c)

    assert torch.allclose(y, y2)


def test_total_count():
    y = torch.randint(
        0,
        5,
        (4, 1, 16, 16),
        dtype=torch.float32,
    )

    c = integral_2d(y)

    n1 = y.sum(dim=(-2, -1))
    n2 = c[..., -1, -1]

    assert torch.allclose(n1, n2)


def test_nonnegative_measure():
    y = torch.rand(2, 1, 8, 8)
    c = integral_2d(y)

    y_rec = mixed_difference_2d(c)

    assert (y_rec >= -1e-6).all()
```

---

# 49. Minimal Experiment YAML

```yaml
experiment: micf_triangle_sha

dataset:
  name: ShanghaiTechA
  crop_size: 256
  output_stride: 16

model:
  backbone: mobilenetv4_conv_small_050
  pretrained: false
  fpn_channels: 32

optimizer:
  name: AdamW
  lr: 0.0001
  weight_decay: 0.0001

training:
  epochs: 1000
  warmup_epochs: 25
  grad_clip: 5.0
  seeds: [41, 42, 43]

variants:
  - local
  - local_cumulative_loss
  - direct_micf

loss:
  type: smooth_l1
```

---

# 50. Output CSV Schema

```text
dataset
seed
variant
params
flops
rf_proxy
mae
rmse
nae
prefix_mae
local_recon_mae
rectangle_mae_small
rectangle_mae_medium
rectangle_mae_large
negative_cell_fraction
negative_mass_ratio
corner_delta_count_gap
train_time
peak_vram
```

---

# 51. Implementation Order

1. implement exact \(Y\);
2. verify \(\sum Y=N\);
3. implement cumulative target;
4. verify mixed-difference inverse;
5. run A;
6. run B;
7. run C;
8. run 3 seeds;
9. inspect positional and RF diagnostics;
10. decide whether direct MICF survives;
11. add validity;
12. add orientation balancing;
13. add integral context;
14. run capacity/RF sweeps;
15. add QNRF/NWPU/ZIP.

---

# 52. Core Hypothesis in One Sentence

\[
\boxed{
\text{Spatial integration may convert a hard sparse local target into a smoother globally structured target that tiny models learn more efficiently, but only until the resulting non-local context requirement exceeds their representational capacity.}
}
\]

This trade-off is the real research problem.

---

# 53. Final Preferred Stack

```text
point annotations
    ↓
exact local count map Y
    ↓
exact cumulative target C
    ↓
tiny MobileNet/FPN features
    ↓
optional 4-direction normalized integral feature context
    ↓
direct cumulative field C_hat
    ↓
field loss
+ mixed-difference validity
+ orientation-balanced augmentation
    ↓
count = C_hat[-1,-1]
and/or
Y_hat = Δxy C_hat
```

Always compare against:

```text
Y_hat + local loss
Y_hat + cumulative loss
softplus(Y_hat) -> deterministic integral
```

Without these controls, MICF is not scientifically identifiable.

---

# 54. Final Decision Logic

```text
C > B > A
    -> direct representation survives

B ≈ C > A
    -> pivot to integral-domain loss

B > C
    -> cumulative loss survives, direct MICF fails

A >= B,C
    -> kill integral-domain direction
```

If direct MICF survives:

```text
+ validity
+ orientation balancing
+ integral context
+ capacity sweep
+ RF sweep
+ multiple datasets
+ ZIP / external baselines
```

Only after this evidence should MICF become the full paper method.
