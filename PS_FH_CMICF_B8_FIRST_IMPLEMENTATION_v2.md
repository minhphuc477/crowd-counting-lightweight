# PS-FH-CMICF — Corrected B8-First Implementation Plan

## Preconditioned Sobolev Finite-Horizon Cumulative Measure Integral Counting Field

**Base model:** B8 FH-CMICF  
**Canonical starting geometry:** `output_stride=16`, `finite_horizon=4`  
**Repository:** `minhphuc477/crowd-counting-lightweight`  
**Branch:** `MICF`

---

# 0. Correction

The proposed method must **develop from B8**, not from B9.

B8 is currently the strongest FH-CMICF formulation:

\[
s=16,\qquad K=4,\qquad \text{physical FH span}=64\text{ px}.
\]

Its key results on ShanghaiTech Part A are:

- Direct MAE: **115.22**
- Controlled tiled MAE: **103.23**
- Practical tiled MAE: **105.98**
- Direct negative-mass ratio: **1.47%**
- Direct violation rate: **3.78%**
- Direct–Controlled gap: **11.99**

B9 is **not** the base model. It is only a diagnostic stress test:

\[
s=4,\qquad K=16.
\]

B9 shows that naively increasing the cumulative chart from \(4\times4\) to \(16\times16\) while preserving a 64-pixel physical horizon is unstable.

Therefore the development path is:

```text
B8 FH-CMICF
    ↓
PS-FH-CMICF @ s16, K4
    ↓
mechanism verification
    ↓
increase spatial resolution while KEEPING K SMALL
    ↓
s8, K4
    ↓
s4, K4 if needed
```

Do **not** use `s4,K16` as the canonical PS-FH-CMICF model.

---

# 1. Base formulation: B8 FH-CMICF

B8 predicts finite-horizon cumulative charts and composes them exactly into a global cumulative field.

For exact cell counts

\[
Y_{ij}
=
\#\left\{
p_n:
\left\lfloor \frac{y_n+0.5}{s}\right\rfloor=i,\;
\left\lfloor \frac{x_n+0.5}{s}\right\rfloor=j
\right\},
\]

the cumulative field is

\[
C_{ij}
=
\sum_{a\le i,b\le j}Y_{ab}.
\]

The exact inverse is

\[
Y_{ij}
=
\Delta_{xy}C_{ij}
=
C_{ij}
-C_{i-1,j}
-C_{i,j-1}
+C_{i-1,j-1}.
\]

Validity requires

\[
\Delta_{xy}C\ge0.
\]

B8 uses \(K=4\), giving a short finite cumulative chart.

---

# 2. What is wrong with B8

The diagnostics show three real limitations.

## 2.1 Phase dependence

Within each \(4\times4\) local chart:

\[
(u,v)\in\{0,1,2,3\}^2.
\]

Negative recovered mass increases strongly with cumulative phase distance:

\[
d=u+v.
\]

Measured correlations are approximately:

\[
r\approx0.90-0.93
\]

across scales.

The terminal phase \((3,3)\) is consistently worst.

Safe conclusion:

> B8 exhibits systematic within-horizon cumulative phase bias.

---

## 2.2 Boundary sensitivity

Regions crossing FH boundaries are worse than matched interior regions.

At \(1\times\):

- clean-interior GT MAE: **0.6976**
- boundary-straddling GT MAE: **0.7911**

This effect exists without any image rescaling.

---

## 2.3 Partition-origin sensitivity

Grid-offset regions are worse than grid-aligned regions.

At \(1\times\):

- aligned 64-px region GT MAE: **2.0354**
- offset-junction GT MAE: **2.2684**

Therefore the fixed local chart origin is not a neutral coordinate choice.

---

# 3. What B9 contributes

B9 is only evidence that one obvious scaling strategy is wrong.

B9:

\[
s=4,\qquad K=16.
\]

Results:

- Direct MAE: **221.82**
- Controlled tiled MAE: **106.25**
- Practical tiled MAE: **143.55**
- Direct negative mass: **12.49%**
- Direct violation rate: **16.18%**

The correct interpretation is:

> keeping the same physical 64-pixel FH span while increasing the cumulative chart from \(K=4\) to \(K=16\) makes the learned cumulative subproblem much harder.

B9 must appear only as:

- diagnostic evidence;
- a warning against large-\(K\) cumulative charts;
- motivation for operator-aware optimization.

It is **not** the model from which PS-FH-CMICF is developed.

---

# 4. PS-FH-CMICF: changes applied directly to B8

The canonical model is:

\[
\boxed{s=16,\ K=4}
\]

with the same:

- MobileNetV4 Small 0.5 backbone;
- Additive FPN;
- directional integral context;
- neck width 32;
- ImageNet pretraining;
- optimizer;
- crop size;
- training schedule.

Only the cumulative mechanism and optimization are changed.

---

# 5. Component A — Strict-local FH cumulative head

Current B8 only makes the integral pooling finite-horizon.

After pooling, learned fusion and the task head still operate on the reassembled full feature map.

Change this to:

```text
Backbone/FPN
    ↓
P16 features
    ↓
partition into K×K blocks
    ↓
directional integral context inside each block
    ↓
coord injection inside each block
    ↓
DW3×3 + normalization + output head inside each block
    ↓
local cumulative chart C_b
    ↓
exact Δxy
    ↓
exact global composition
```

The visual backbone remains global.

Only FH-specific cumulative operations become block-local.

This directly targets the measured partition/origin instability.

---

# 6. Component B — Fractionally preconditioned cumulative loss

For one \(K\times K\) chart, define the prefix operator:

\[
T_K=A_K\otimes A_K
\]

with

\[
A_K=
\begin{bmatrix}
1&0&\cdots&0\\
1&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
1&1&\cdots&1
\end{bmatrix}.
\]

Take:

\[
T_K=U\Sigma V^\top.
\]

Define the fixed preconditioner

\[
P_\alpha
=
U\Sigma^{-\alpha}U^\top.
\]

Default:

\[
\boxed{\alpha=0.5}.
\]

Use:

\[
\mathcal L_{\mathrm{PC}}
=
\left\|
P_\alpha
\frac{\hat C-C}{\max(N,1)}
\right\|_2^2.
\]

For B8, this does not change the architecture or inference cost.

Its purpose is to reduce the spectral imbalance of the cumulative residual.

---

# 7. Component C — Sobolev mixed-difference supervision

The model still predicts \(C\) directly.

Recover:

\[
\hat Y=\Delta_{xy}\hat C.
\]

Add:

\[
\mathcal L_{\mathrm{Sob}}
=
\operatorname{SmoothL1}
(
\Delta_{xy}\hat C,\,
Y
).
\]

Use a balanced foreground/background average so zero cells do not dominate.

This is derivative supervision of a cumulative field, not a local-\(Y\) prediction head.

---

# 8. Component D — Augmented-Lagrangian validity

The recovered measure must satisfy:

\[
\Delta_{xy}\hat C\ge0.
\]

Define:

\[
g
=
\mathbb E
\left[
\frac{
\sum \max(-\Delta_{xy}\hat C,0)
}{
\max(N,1)
}
\right].
\]

Use:

\[
\mathcal L_{\mathrm{AL}}
=
\lambda g+\frac{\rho}{2}g^2.
\]

Update:

\[
\lambda
\leftarrow
\operatorname{clip}
(
\lambda+\rho g,\,
0,\lambda_{\max}
).
\]

This replaces the fixed B8 validity weight.

---

# 9. Final PS-FH-CMICF loss

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{PC}}
+
\lambda_S\mathcal L_{\mathrm{Sob}}
+
\lambda_N\mathcal L_N
+
\mathcal L_{\mathrm{AL}}
}
\]

with:

\[
\mathcal L_N
=
\operatorname{SmoothL1}
\left(
\frac{\hat N-N}{\max(N,1)},0
\right).
\]

Initial values:

```yaml
precondition_alpha: 0.5
lambda_sobolev: 1.0
lambda_count: 1.0
al_rho: 1.0
al_dual_init: 0.0
al_dual_max: 100.0
norm_eps: 1.0
```

---

# 10. Canonical config

Create:

`configs/pilot_micf/psfh_b8_k4.yaml`

```yaml
augmentation:
  flip_prob: 0.5
  scale_range:
    - 0.7
    - 1.3
  vflip_prob: 0.5

dataset:
  coordinate_base: 0
  crop_size: 256
  image_mean:
    - 0.5
    - 0.5
    - 0.5
  image_std:
    - 0.5
    - 0.5
    - 0.5
  name: sha
  part: part_A
  root: ./data/ShanghaiTech

experiment:
  description: >
    PS-FH-CMICF developed directly from B8:
    strict-local K4 cumulative charts,
    fractional prefix preconditioning,
    Sobolev mixed-difference supervision,
    augmented-Lagrangian 2-increasing constraint.

  model_id: PSFH_B8_K4
  name: ps_fh_cmicf_b8_k4
  save_dir: ./runs/pilot_micf/psfh_b8_k4
  seed: 42

model:
  backbone: mobilenetv4_conv_small_050.e3000_r224_in1k
  pretrained: true
  neck_width: 32

  context_dilations:
    - 1
    - 2
    - 3

  use_integral_context: true
  context_type: directional

  head_type: cumulative
  extent_aware: true

  # SAME B8 geometry.
  output_stride: 16
  finite_horizon: 4

  # New PS-FH switch.
  fh_strict_local: true

  eps_d: 1.0e-08

loss:
  mode: ps_fh_cmicf

  precondition_alpha: 0.5
  precondition_sv_floor: 1.0e-08

  lambda_sobolev: 1.0
  sobolev_beta: 1.0

  lambda_count: 1.0

  al_rho: 1.0
  al_dual_init: 0.0
  al_dual_max: 100.0

  norm_eps: 1.0

optimizer:
  name: AdamW
  lr: 0.0001
  backbone_lr_scale: 0.1
  weight_decay: 0.0001
  grad_clip: 5.0

schedule:
  epochs: 1000
  warmup_epochs: 25

training:
  amp: true
  batch_size: 16
  drop_last: true
  evaluate_every: 5
  num_workers: 0
  pin_memory: false
```

---

# 11. Code changes

Use the same implementation introduced previously:

## Add

```text
hpc/losses/ps_fh_cmicf.py
```

containing:

- `FractionalPrefixPreconditioner`
- `PSFHCMICFLoss`
- block target partition utilities.

## Modify

```text
hpc/models/micf_lite.py
```

Add:

```python
fh_strict_local: bool = False
```

and a strict-local branch that partitions the feature map **before** DIC fusion and task-head execution.

## Modify

```text
tools/train_micf_pilot.py
```

Add:

- `PSFHCMICFLoss`
- `criterion.to(device)`
- `forward_field_with_aux`
- per-epoch dual update
- PS loss logging
- criterion state checkpointing.

## Modify builders

```text
tools/eval_micf_comprehensive.py
tools/profile_model.py
tools/export_onnx.py
```

to pass:

```python
fh_strict_local=bool(
    m_cfg.get(
        "fh_strict_local",
        False,
    )
)
```

---

# 12. First experiment: B8 vs PS-FH-CMICF

This is the first and most important comparison.

| Model | stride | K | Architecture | Loss |
|---|---:|---:|---|---|
| B8 | 16 | 4 | old FH scope | SmoothL1-C + fixed validity |
| PS-FH-CMICF | 16 | 4 | strict-local FH | preconditioned C + Sobolev + AL |

Everything else must remain matched.

---

# 13. Success criteria on B8 geometry

PS-FH-CMICF must improve the actual measured limitations.

## Counting

Target:

\[
\text{Direct MAE}<115.22.
\]

Preferably:

\[
<100.
\]

## Validity

Target:

\[
\text{negative mass}<1.47\%.
\]

Target:

\[
\text{violation rate}<3.78\%.
\]

## Direct–Tiled stability

Target:

\[
\text{Direct-Controlled gap}<11.99.
\]

## Phase

Current:

\[
r(d,\text{negative mass})\approx0.90-0.93.
\]

Target:

\[
\text{material reduction}.
\]

## Boundary

Boundary-straddling vs clean-interior gap should shrink.

## Partition origin

Grid-offset vs aligned gap should shrink.

The method should not be accepted merely because crop MAE improves.

---

# 14. What happens after B8 is fixed

Only after PS-FH-CMICF succeeds on `s16,K4` should the model be made higher resolution.

The correct scaling principle is:

> **increase spatial resolution while keeping cumulative chart size small.**

Therefore:

## Stage 2

\[
\boxed{s=8,\ K=4}
\]

Physical cumulative horizon:

\[
8\times4=32\text{ px}.
\]

This doubles spatial output resolution in each dimension while preserving a short \(4\times4\) cumulative chart.

## Stage 3

Only if needed:

\[
\boxed{s=4,\ K=4}
\]

Physical cumulative horizon:

\[
4\times4=16\text{ px}.
\]

The backbone/FPN can still provide visual context larger than the cumulative horizon.

---

# 15. Do not preserve 64-pixel physical FH span by increasing K

Do **not** use:

\[
s=8,K=8
\]

or

\[
s=4,K=16
\]

as the default scaling strategy.

B9 shows that preserving physical horizon by increasing \(K\) can severely worsen the cumulative learning problem.

The cumulative horizon and visual receptive field are different concepts.

The model can use wide backbone context while keeping the integral operator local.

---

# 16. Accuracy path to SOTA

Once the B8 mechanism is fixed:

```text
B8 s16 K4
    ↓
PS-FH-CMICF s16 K4
    ↓
PS-FH-CMICF s8 K4
    ↓
PS-FH-CMICF s4 K4 if needed
```

Only then add accuracy improvements that preserve the cumulative contribution:

1. exact multi-scale range-count supervision;
2. chart-origin invariance if boundary bias remains;
3. scale-measure consistency if cross-scale mismatch remains;
4. stronger efficient carrier for a Base model;
5. keep a tiny Lite model for accuracy-efficiency Pareto.

Do not add generic attention/MoE before this chain is complete.

---

# 17. B9's exact role in the paper

B9 should appear only in a diagnostic/ablation table:

| Model | s | K | Physical span | Direct MAE | Neg mass |
|---|---:|---:|---:|---:|---:|
| B8 | 16 | 4 | 64 px | 115.22 | 1.47% |
| B9 | 4 | 16 | 64 px | 221.82 | 12.49% |

Interpretation:

> Matching physical FH span is insufficient; cumulative chart resolution itself materially affects learnability and validity.

Then the proposed method develops from B8 and scales with **small K**.

---

# 18. Run command

Canonical B8-derived PS-FH-CMICF:

```powershell
$env:PYTHONPATH="."

.venv\Scripts\python tools/train_micf_pilot.py `
  --config configs/pilot_micf/psfh_b8_k4.yaml `
  --device cuda
```

Evaluation:

```powershell
$env:PYTHONPATH="."

.venv\Scripts\python tools/eval_micf_comprehensive.py `
  --checkpoint runs/pilot_micf/psfh_b8_k4/best.pt `
  --config configs/pilot_micf/psfh_b8_k4.yaml `
  --device cuda `
  --halo 64
```

Then rerun:

```text
16-phase audit
boundary audit
partition-origin audit
scale-consistency audit
```

---

# 19. Final development path

The corrected roadmap is:

```text
B8 FH-CMICF
s16 K4
best current FH model
        ↓
PS-FH-CMICF
s16 K4
same geometry
        ↓
prove:
phase ↓
boundary gap ↓
negative mass ↓
Direct-Tiled gap ↓
MAE ↓
        ↓
PS-FH-CMICF
s8 K4
        ↓
PS-FH-CMICF
s4 K4 if required
        ↓
regional supervision / chart invariance
        ↓
Lite + Base variants
        ↓
SHA + QNRF + NWPU + JHU
        ↓
lightweight SOTA / absolute SOTA attempt
```

The method is therefore **a direct development of B8**, with B9 retained only as evidence about what *not* to do when increasing spatial resolution.
---

# 20. Implementation cautions that are now mandatory

These are not optional notes. They are part of the canonical PS-FH-CMICF implementation protocol.

## 20.1 Sobolev sparsity control

The recovered local measure is sparse:

\[
Y_{ij}=0
\]

for most cells.

Therefore this is forbidden:

```python
F.smooth_l1_loss(
    pred_y,
    target_y,
)
```

over the whole tensor without stratification.

Use:

\[
\mathcal L_{\mathrm{Sob}}
=
\frac12
\mathbb E_{Y>0}
\left[
\rho(\hat Y-Y)
\right]
+
\frac12
\mathbb E_{Y=0}
\left[
\rho(\hat Y-Y)
\right].
\]

If one stratum is absent, use the available stratum alone.

### Required logging

Add:

```text
sobolev_pos_loss
sobolev_zero_loss
positive_cell_fraction
zero_cell_fraction
```

Optional but strongly recommended:

```text
sobolev_pos_grad_norm
sobolev_zero_grad_norm
```

The purpose is to verify that the local derivative supervision does not collapse to a trivial near-zero solution.

---

# 21. Improved Sobolev implementation

Replace the simple `_balanced_smooth_l1` helper with:

```python
def balanced_sobolev_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float,
):
    per = F.smooth_l1_loss(
        pred,
        target,
        beta=beta,
        reduction="none",
    )

    pos_mask = target > 0
    zero_mask = ~pos_mask

    zero = pred.new_zeros(())

    if bool(pos_mask.any()):
        pos_loss = per[pos_mask].mean()
    else:
        pos_loss = zero

    if bool(zero_mask.any()):
        zero_loss = per[zero_mask].mean()
    else:
        zero_loss = zero

    if bool(pos_mask.any()) and bool(zero_mask.any()):
        total = 0.5 * (
            pos_loss
            + zero_loss
        )
    elif bool(pos_mask.any()):
        total = pos_loss
    else:
        total = zero_loss

    stats = {
        "sobolev_pos_loss":
            float(
                pos_loss.detach().item()
            ),

        "sobolev_zero_loss":
            float(
                zero_loss.detach().item()
            ),

        "positive_cell_fraction":
            float(
                pos_mask.float()
                .mean()
                .detach()
                .item()
            ),

        "zero_cell_fraction":
            float(
                zero_mask.float()
                .mean()
                .detach()
                .item()
            ),
    }

    return total, stats
```

Then in `PSFHCMICFLoss.forward` use:

```python
sobolev_loss, sob_stats = (
    balanced_sobolev_smooth_l1(
        pred_y_blocks,
        target_y_blocks,
        beta=self.sobolev_beta,
    )
)
```

and merge `sob_stats` into the returned components.

---

# 22. Augmented-Lagrangian diagnostics

The augmented Lagrangian is:

\[
\mathcal L_{\mathrm{AL}}
=
\lambda g
+
\frac{\rho}{2}g^2.
\]

with:

\[
g
=
\mathbb E
\left[
\frac{
\sum \max(-\Delta C,0)
}{
\max(N,1)
}
\right].
\]

Do not tune \(\rho\) blindly.

Start with:

```yaml
al_rho: 1.0
al_dual_init: 0.0
al_dual_max: 100.0
```

Update:

\[
\lambda_{t+1}
=
\operatorname{clip}
(
\lambda_t+\rho g_t,\,
0,\lambda_{\max}
).
\]

### Required logging

Every evaluation epoch log:

```text
ps_constraint
ps_dual_lambda
ps_al_rho
ps_aug_lagrangian
negative_mass_ratio
violation_rate
grad_norm_before_clip
grad_norm_after_clip
clip_triggered
```

The dual variable must not be a hidden internal state.

---

# 23. Adaptive AL rule

Do not change \(\rho\) during the first canonical run unless optimization clearly fails.

If adaptive control is later needed, use a deterministic rule rather than manual trial-and-error.

Maintain an EMA:

\[
\bar g_t
=
\beta
\bar g_{t-1}
+
(1-\beta)g_t,
\qquad
\beta=0.9.
\]

Every 25 evaluation epochs:

### Increase \(\rho\)

If:

\[
\bar g_t
>
0.9\bar g_{t-25},
\]

meaning the constraint barely improves:

```python
rho = min(
    rho * 2.0,
    rho_max,
)
```

### Decrease \(\rho\)

If all are true:

```text
gradient clipping triggers in >50% of recent batches
constraint oscillates instead of decreasing
training loss becomes visibly unstable
```

then:

```python
rho = max(
    rho * 0.5,
    rho_min,
)
```

Suggested bounds:

```yaml
al_rho_min: 0.125
al_rho_max: 8.0
```

This adaptive rule is **not enabled in the first paper experiment**.

It is a fallback implementation only.

---

# 24. Gradient diagnostics

The new objective contains terms with different geometries:

\[
L_{\mathrm{PC}},
\quad
L_{\mathrm{Sob}},
\quad
L_N,
\quad
L_{\mathrm{AL}}.
\]

Their scalar magnitudes alone do not reveal their actual optimization influence.

For a small diagnostic subset, compute gradient norms with respect to the directly predicted local cumulative charts:

\[
G_q
=
\left\|
\frac{\partial L_q}
{\partial C_{\mathrm{blocks}}}
\right\|_2.
\]

Required diagnostic terms:

```text
grad_pc
grad_sobolev
grad_count
grad_al
```

Do this only:

```text
every 25 epochs
on one training batch
without optimizer.step()
```

to avoid large overhead.

### Helper

```python
def grad_norm_wrt(
    loss: torch.Tensor,
    tensor: torch.Tensor,
) -> float:
    grad = torch.autograd.grad(
        loss,
        tensor,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]

    if grad is None:
        return 0.0

    return float(
        grad.detach()
        .float()
        .norm()
        .item()
    )
```

Interpretation:

- `grad_sobolev >> grad_pc` by orders of magnitude: Sobolev term dominates.
- `grad_pc >> grad_sobolev`: preconditioned field metric dominates.
- `grad_al` explodes as `lambda` rises: AL schedule is too aggressive.
- `grad_count` dominates everything: whole-count term is masking field learning.

Do not use automatic gradient balancing in the first method.

Only diagnose.

---

# 25. Strict-local implementation: vectorize blocks, never loop

Do not use Python loops over blocks.

The existing repository already has the correct transformation:

\[
[B,C,H,W]
\rightarrow
[B\,n_hn_w,C,K,K].
\]

Canonical implementation:

```python
blocks = (
    x.view(
        B,
        C,
        nh,
        k,
        nw,
        k,
    )
    .permute(
        0,
        2,
        4,
        1,
        3,
        5,
    )
    .contiguous()
    .view(
        B * nh * nw,
        C,
        k,
        k,
    )
)
```

Then all Conv/normalization/head operations run in one vectorized call.

Avoid:

```text
for block in blocks
torch.chunk loops
nested loops
```

`unfold/fold` is unnecessary for non-overlapping FH blocks and introduces avoidable memory traffic.

---

# 26. Critical normalization confound

Strict-localizing the head changes normalization behavior.

Current directional context uses:

```text
BatchNorm2d
```

and the task head uses GroupNorm.

When blocks become batch elements:

\[
[B,C,H,W]
\rightarrow
[Bn_hn_w,C,K,K],
\]

BatchNorm statistics change.

Therefore an observed improvement could come partly from changed normalization statistics rather than the finite-horizon architecture alone.

This must be controlled.

---

# 27. Canonical normalization choice

For the proposed PS-FH-CMICF method, use **block-independent channel normalization** inside the FH-specific path.

Recommended:

> **GroupNorm with one group per suitable channel partition**, applied independently to each \(K\times K\) chart tensor.

It has no running statistics and does not depend on the total number of image blocks at inference.

Use the repository's existing:

```python
make_group_norm(channels)
```

where possible.

Do not use BatchNorm inside the strict-local cumulative module.

The backbone may keep its pretrained normalization unchanged.

---

# 28. Add strict-local DIC variant

In:

`hpc/models/integral_context.py`

add:

```python
from .blocks import make_group_norm
```

Create:

```python
class StrictLocalDirectionalIntegralContext(nn.Module):
    """
    Directional integral context intended for already-partitioned
    KxK chart batches.

    No global spatial statistics.
    """

    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.use_residual = use_residual

        in_channels = channels * 5

        self.reduce_conv = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.reduce_norm = (
            make_group_norm(
                channels
            )
        )

        self.reduce_act = nn.SiLU(
            inplace=True
        )

        self.dw_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.dw_norm = (
            make_group_norm(
                channels
            )
        )

        self.dw_act = nn.SiLU(
            inplace=True
        )

        self.project = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

        self.dropout = (
            nn.Dropout2d(
                dropout
            )
            if dropout > 0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tl = normalized_integral_tl(x)
        tr = normalized_integral_tr(x)
        bl = normalized_integral_bl(x)
        br = normalized_integral_br(x)

        z = torch.cat(
            [x, tl, tr, bl, br],
            dim=1,
        )

        z = self.reduce_conv(z)
        z = self.reduce_norm(z)
        z = self.reduce_act(z)

        z = self.dw_conv(z)
        z = self.dw_norm(z)
        z = self.dw_act(z)

        z = self.project(z)
        z = self.dropout(z)

        if self.use_residual:
            return x + z

        return z
```

---

# 29. Modify MICFLite for the strict-local context

Import:

```python
from .integral_context import (
    AxialIntegralContext,
    DirectionalIntegralContext,
    StrictLocalDirectionalIntegralContext,
)
```

Add constructor option:

```python
fh_local_norm: str = "group",
```

Store:

```python
self.fh_local_norm = str(
    fh_local_norm
).lower()
```

When building the context module:

```python
if (
    self.finite_horizon is not None
    and self.fh_strict_local
):
    self.context_module = (
        StrictLocalDirectionalIntegralContext(
            channels=self.neck_width
        )
    )
elif self.context_type == "directional":
    self.context_module = (
        DirectionalIntegralContext(
            channels=self.neck_width
        )
    )
```

This makes the normalization behavior explicit and reproducible.

---

# 30. Required normalization ablation

This is one small but necessary control.

Do not run a large sweep.

Compare only:

| ID | Model | FH scope | Context normalization |
|---|---|---|---|
| B8 | original | prefix only local | BatchNorm |
| PS-NormCtrl | strict-local | all FH learned ops local | BatchNorm |
| PS-FH | strict-local | all FH learned ops local | GroupNorm |

Purpose:

- `B8 -> PS-NormCtrl` isolates strict-local scope approximately.
- `PS-NormCtrl -> PS-FH` isolates normalization stability.

This control can be short-trained first.

It is not a new contribution.

---

# 31. Preconditioner alpha policy

Canonical:

\[
\boxed{\alpha=0.5}.
\]

Reason:

\[
\kappa^2
\rightarrow
\kappa
\]

for the idealized quadratic cumulative residual.

Do not sweep alpha before the canonical B8-derived run.

Only consider:

\[
\alpha\in\{0.25,0.75\}
\]

if the following occurs:

```text
preconditioned field loss converges
Sobolev loss stagnates
phase/validity metrics do not improve
task MAE plateaus
```

Do not select alpha on the test set.

---

# 32. Preconditioner diagnostics

At initialization print:

```text
finite_horizon K
alpha
min singular value
max singular value
prefix condition number
effective quadratic condition number
```

Example:

```python
p = criterion.preconditioner

print(
    "PS-FH preconditioner | "
    f"K={p.k} | "
    f"alpha={p.alpha:.2f} | "
    f"kappa(T)={p.prefix_condition_number:.3f} | "
    f"kappa_eff={p.quadratic_condition_number:.3f}"
)
```

Also write the values into the checkpoint metadata.

---

# 33. Revised canonical B8-first config

Use:

```yaml
model:
  backbone: mobilenetv4_conv_small_050.e3000_r224_in1k
  pretrained: true

  neck_width: 32

  context_dilations:
    - 1
    - 2
    - 3

  use_integral_context: true
  context_type: directional

  head_type: cumulative
  extent_aware: true

  output_stride: 16
  finite_horizon: 4

  fh_strict_local: true
  fh_local_norm: group

  eps_d: 1.0e-08

loss:
  mode: ps_fh_cmicf

  precondition_alpha: 0.5
  precondition_sv_floor: 1.0e-08

  lambda_sobolev: 1.0
  sobolev_beta: 1.0

  lambda_count: 1.0

  al_rho: 1.0
  al_dual_init: 0.0
  al_dual_max: 100.0

  norm_eps: 1.0
```

Everything else remains exactly matched with B8.

---

# 34. Canonical experiment sequence

Do not run a broad model search.

Run only:

```text
E0 = existing B8
E1 = strict-local scope + normalization control
E2 = PS-FH-CMICF full @ s16,K4
```

The full proposed model is E2.

The acceptance criteria are:

```text
Direct MAE ↓
Direct-Tiled gap ↓
negative mass ↓
violation rate ↓
phase-distance correlation ↓
boundary gap ↓
partition-origin gap ↓
```

Then proceed to:

```text
E3 = PS-FH-CMICF @ s8,K4
```

Do not move to higher resolution before E2 demonstrates mechanism improvement.

---

# 35. Required training log schema

Each evaluation entry should contain at least:

```json
{
  "epoch": 100,
  "loss": 0.0,

  "ps_pc_loss": 0.0,
  "ps_sobolev_loss": 0.0,
  "sobolev_pos_loss": 0.0,
  "sobolev_zero_loss": 0.0,

  "ps_count_loss": 0.0,

  "ps_constraint": 0.0,
  "ps_dual_lambda": 0.0,
  "ps_al_rho": 1.0,
  "ps_aug_lagrangian": 0.0,

  "positive_cell_fraction": 0.0,

  "grad_norm_before_clip": 0.0,
  "grad_norm_after_clip": 0.0,
  "clip_trigger_rate": 0.0,

  "mae_crop": 0.0,
  "mae_full_direct": 0.0,
  "mae_full_tiled": 0.0,

  "negative_mass_ratio": 0.0,
  "violation_rate": 0.0
}
```

Do not rely only on total training loss.

---

# 36. Final implementation caution summary

The proposed model is valid only if all of the following are respected:

1. Sobolev derivative supervision is foreground/background balanced.
2. FH blocks are vectorized with reshape/permute, not Python loops.
3. Strict-local context does not use full-image BatchNorm statistics.
4. The preconditioner remains fixed and non-learned.
5. `alpha=0.5` is the canonical first choice.
6. Augmented-Lagrangian dual state is logged and checkpointed.
7. Negative mass uses the canonical:
   \[
   M^- / M^+.
   \]
8. B8 `s16,K4` is the development base.
9. B9 remains diagnostic only.
10. Higher resolution is attempted only after the B8-derived mechanism is verified.
