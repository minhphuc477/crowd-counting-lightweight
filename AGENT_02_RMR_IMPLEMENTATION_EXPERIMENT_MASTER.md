# AGENT 02 — RMR-Count Full Implementation & Experiment Master Specification

> **Role:** implementation / experiment / evaluation agent.  
> **Target:** implement and execute the registered RMR-Count CVPR experiment without changing the scientific question.  
> **Research source of truth:** `AGENT_01_CVPR_RMR_RESEARCH_PAPER_MASTER.md`.  
> **Implementation source of truth:** this file.  
> **Canonical hardware for the project:** one NVIDIA T4 16 GB unless a separate experiment explicitly states otherwise.  
> **Status of embedded reference code:** syntax checked and unit-tested; **11 tests pass** in the generated reference tree.

---

# 0. Mission

Implement and evaluate one central mechanism:

\[
\boxed{
Y^{(t)}
\rightarrow
AY^{(t)}
\rightarrow
\delta^{(t)}=AY^{(t)}-b
\rightarrow
A^\top\left(\delta^{(t)}/|R|\right)
\rightarrow
Y^{(t+1)}
}
\]

where:

- \(Y^{(t)}\) is a fine, non-negative stride-4 count measure;
- \(A\) is the exact rectangular regional-sum operator;
- \(b\) is separately inferred visual regional count evidence;
- \(A^\top\) is the exact adjoint of regional summation;
- a small bounded local preconditioner may modulate the exact correction;
- the final model must remain native ultra-lightweight.

The agent must **not** turn this into a module-collection project.

---

# 1. Non-negotiable rules

1. Do not add attention, transformer, MoE, teacher/distillation, ASPP, extra heavy branches, or a new loss merely because B5 is weak.
2. Do not tune on the test set.
3. Use validation only for learning-rate/model-selection decisions.
4. Freeze the selected LR/protocol before final repeated-seed comparisons.
5. B0–B5 must share the same carrier, data split, primary supervision, output stride, augmentation, optimizer family, and checkpoint-selection rule.
6. Report exact measured Params, FLOPs/MACs convention, latency, p95 latency, and peak memory.
7. The strongest geometry test is **B3b vs B5**, not B3a vs B5 alone.
8. Start regional scales at:
   \[
   \boxed{\{32,64,128\}\text{ px}}
   \]
   and add 16 px only if oracle/predicted diagnostics justify it.
9. All RMR/learned-refinement variants use solver warm-up/ramp and bounded step size.
10. If results fail the registered kill rules, stop or reframe; do not “rescue” the model by adding arbitrary components.

---

# 2. Registered experiment matrix

| ID | config / variant | Regional evidence | Projection / refinement | Purpose |
|---|---|---|---|---|
| **B0** | `direct` | none | none | carrier baseline |
| **B1** | `region_loss` | training-only regional map loss | none | regional supervision control |
| **B2** | `region_aux` | regional head \(b_R\) | none | auxiliary-head control |
| **B3a** | `local_refine` | none | local 3×3 learned refiner | extra local capacity control |
| **B3b** | `learned_project` | same \(b_R,\delta_R,\mathcal R\) | learned region-membership projection | strongest geometry control |
| **B4** | `rmr`, `T=1` | same \(b_R\) | exact \(A^\top\), one step | one-step RMR |
| **B5** | `rmr`, `T=2` | same \(b_R\) | exact \(A^\top\)+local preconditioner | canonical RMR |

The two key comparisons are:

\[
\boxed{B5\ \text{vs}\ B3b}
\]

for the exact-geometry claim, and:

\[
\boxed{B5\ \text{vs}\ B3a}
\]

for the “operator vs additional local neural capacity” efficiency claim.

---

# 3. Fine counting measure

For output stride \(s=4\), rasterize point annotations directly:

\[
Y^{gt}_{ij}
=
\#\left\{
p_n:
\left\lfloor
\frac{y_n+0.5}{4}
\right\rfloor=i,\quad
\left\lfloor
\frac{x_n+0.5}{4}
\right\rfloor=j
\right\}.
\]

Required invariants:

\[
Y^{gt}_{ij}\ge0,
\qquad
\sum_{ij}Y^{gt}_{ij}=N.
\]

The model predicts latent logits \(z\) and uses:

\[
\boxed{
Y=\operatorname{softplus}(z)
}
\]

so positivity is guaranteed by construction.

Do not clip \(Y\) after prediction as a substitute for this parameterization.

---

# 4. Carrier architecture

Canonical lightweight carrier:

| stage | stride | channels | blocks |
|---|---:|---:|---:|
| stem | 2 | 16 | 1 |
| \(C_4\) | 4 | 24 | 2 |
| \(C_8\) | 8 | 40 | 3 |
| \(C_{16}\) | 16 | 64 | 2 |
| fused \(F\) | 4 | 32 | lightweight additive fusion |

Local block:

\[
DWConv_{3\times3}
\rightarrow
1\times1\text{ expansion}
\rightarrow
SiLU
\rightarrow
1\times1\text{ projection}
\rightarrow
\text{residual}.
\]

Fusion:

\[
F
=
\phi
\left(
P_4(C_4)
+
U_2(P_8(C_8))
+
U_4(P_{16}(C_{16}))
\right).
\]

No independent learned cumulative head exists.

---

# 5. Regional operator \(A\)

Let:

\[
\mathcal R=\{R_m\}_{m=1}^M.
\]

Define:

\[
(AY)_m
=
\sum_{p\in R_m}Y_p.
\]

The exact cumulative table is:

\[
C_{ij}
=
\sum_{a\le i,b\le j}Y_{ab}.
\]

For half-open rectangle:

\[
R=[y_1,y_2)\times[x_1,x_2),
\]

\[
Q_R(C)
=
C(y_2,x_2)
-C(y_1,x_2)
-C(y_2,x_1)
+C(y_1,x_1).
\]

Thus:

\[
AY=Q(PY).
\]

This is an implementation of known counting algebra, not a novelty claim.

---

# 6. Regional visual evidence \(b_R\)

Compute an integral feature table:

\[
S_F=P(F).
\]

Regional descriptor:

\[
\bar F_R
=
\frac{Q_R(S_F)}{|R|}.
\]

Shared head:

\[
b_R
=
\operatorname{softplus}
H_R(\bar F_R,g_R),
\]

where \(g_R\) encodes region geometry.

Pilot region scales:

\[
\boxed{
32,\ 64,\ 128\text{ image pixels}
}
\]

plus the full image.

Do not add 16 px until M3/M4 diagnostics show it helps.

---

# 7. RMR energy and exact adjoint

Regional consistency energy:

\[
E(Y)
=
\frac12
\left\|
W^{1/2}(AY-b)
\right\|_2^2.
\]

For the unweighted pilot:

\[
W=I.
\]

Gradient:

\[
\boxed{
\nabla_YE
=
A^\top(AY-b).
}
\]

Because region sizes differ, use regional residual density:

\[
d_R
=
\frac{(AY-b)_R}{|R|}.
\]

Coverage-normalized exact field:

\[
\boxed{
r
=
\frac{
A^\top d
}{
A^\top\mathbf1+\epsilon
}.
}
\]

The exact adjoint is implemented by a corner difference buffer followed by 2-D cumulative sums.

Mandatory numerical identity:

\[
\boxed{
\langle AY,e\rangle
=
\langle Y,A^\top e\rangle.
}
\]

Unit-test in float64.

---

# 8. Positive bounded update

The exact field is locally preconditioned:

\[
M^{(t)}
=
M_\theta(F,Y^{(t)},r^{(t)}),
\]

bounded by:

\[
M^{(t)}
\in[m_{\min},m_{\max}]
=
[0.25,1.75].
\]

Update:

\[
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
\]

Then:

\[
Y^{(t+1)}
=
\operatorname{softplus}(z^{(t+1)}).
\]

Step-size bounds:

\[
0<\eta_t<\eta_{\max},
\qquad
\eta_{\max}=0.20,
\qquad
\eta_{\text{init}}=0.05.
\]

Residual field is clipped for stability:

\[
r
\leftarrow
\operatorname{clip}(r,-5,5).
\]

These are pilot defaults, not final hyperparameters.

---

# 9. Training-stability protocol

The risk is large regional residual magnitude early in training, not “gradients passing through \(A^\top\) twice.”

For iterative variants:

### epochs \(0,\ldots,E_w-1\)

\[
\text{solver\_strength}=0.
\]

The fine carrier and regional head learn first.

Default:

\[
E_w=5.
\]

### next \(E_r\) epochs

Linearly ramp:

\[
\text{solver\_strength}:0\rightarrow1.
\]

Default:

\[
E_r=20.
\]

Monitor every epoch:

- mean/max \(|r|\);
- \(\eta_0\);
- gradient clipping rate;
- fraction \(z<-10\);
- validation MAE/RMSE;
- iteration-wise regional GT MAE;
- iteration-wise disagreement \(|AY^{(t)}-b|\).

Since \(Y=\operatorname{softplus}(z)\), negative \(Y\) is impossible. The actual saturation risk is \(z\ll0\), where:

\[
\sigma(z)\approx0.
\]

---

# 10. Fair controls

## B3a — local CNN

B3a receives only \(F,Y\) and uses local 3×3 spatial mixing. It tests whether extra local capacity alone explains RMR gains.

Do **not** claim that B3a has the same regional scope as RMR.

## B3b — learned regional projection

B3b receives the same:

\[
b_R,\quad
\delta_R=(AY-b)_R,\quad
\mathcal R,\quad
p\in R
\]

as RMR.

Learn:

\[
\pi_{R,p}
=
\operatorname{softmax}_{p\in R}s_\theta(F_p,Y_p).
\]

Then:

\[
r^{learn}_p
=
\operatorname{Avg}_{R\ni p}
\left[
\delta_R\pi_{R,p}
\right].
\]

This is intentionally a strong control. It may have higher latency than RMR; report that rather than weakening it.

If B3b wins, do not immediately conclude “linear operators fail.” Check energy choice, overlap, residual correlations, \(b_R\) quality, and preconditioning.

---

# 11. Losses

The core paper is not a new-loss paper.

Shared fine loss:

\[
L_{\text{cell}}
=
\frac12
\mathbb E_{Y^{gt}>0}\rho(Y-Y^{gt})
+
\frac12
\mathbb E_{Y^{gt}=0}\rho(Y-Y^{gt}).
\]

Global auxiliary stabilization:

\[
L_N
=
\rho(
\log(1+\hat N)-\log(1+N)
).
\]

B1 training-only regional loss:

\[
L_{\text{region-map}}
=
\operatorname{mean}_R
\rho(
(AY)_R-N_R
).
\]

Regional head:

\[
L_{\text{region-head}}
=
\operatorname{mean}_R
\rho(
b_R-N_R
).
\]

Scale families must be averaged equally so the many small windows do not dominate.

Default:

\[
L
=
L_{\text{cell}}
+0.1L_N
+0.2L_R
\]

with the relevant regional term only for variants that use it.

Final carrier may later use one established strong point-supervised loss for **all** matched variants, but do not change supervision between B0–B5.

---

# 12. Regional-head diagnostics

Do **not** compare 16-px regional MAE directly to whole-image MAE.

Report:

\[
NMAE_R
=
\frac{|b_R-N_R|}
{\max(N_R,1)}.
\]

Also stratify true regional counts:

- 0
- 1
- 2–4
- 5–9
- 10+

For each region scale report predicted and oracle diagnostics.

### Oracle regional evidence

Replace:

\[
b_R
\leftarrow
N_R^{gt}
\]

during evaluation only.

This gives solver headroom.

### Shuffled regional evidence

Shuffle \(b_R\) **within each scale family**.

If performance barely changes, the solver is not using meaningful regional evidence.

---

# 13. Region-scale selection

Run:

\[
\{32\},
\quad
\{32,64\},
\quad
\boxed{\{32,64,128\}},
\quad
\{16,32,64,128\}.
\]

For each scale set:

1. predicted \(b_R\);
2. oracle \(b_R\);
3. regional MAE/NMAE;
4. final MAE/RMSE.

Decision for 16 px:

- oracle 16 does not help → remove;
- oracle helps but predicted 16 hurts → regional predictor bottleneck;
- both help → retain.

---

# 14. Evaluation protocol

Primary:

\[
MAE
=
\frac1n\sum_i|\hat N_i-N_i|,
\]

\[
RMSE
=
\sqrt{\frac1n\sum_i(\hat N_i-N_i)^2}.
\]

Also:

- NAE;
- Bias;
- GAME(0–3);
- regional MAE/NMAE by scale;
- regional MAE by true-count strata;
- iteration-wise regional GT error;
- iteration-wise predicted regional disagreement;
- direct vs controlled tiled;
- direct vs practical tiled;
- Params;
- FLOPs/MACs;
- latency mean/p50/p95;
- FPS;
- peak allocated GPU memory.

Do not select checkpoints by test MAE.

---

# 15. Direct/tiled diagnostics

For direct prediction \(\hat N_i^D\) and tiled prediction \(\hat N_i^T\):

\[
D_{\rm abs}
=
\frac1M
\sum_i
|\hat N_i^D-\hat N_i^T|,
\]

\[
D_{\rm norm}
=
\frac1M
\sum_i
\frac{
|\hat N_i^D-\hat N_i^T|
}{
\max(N_i,1)
}.
\]

Use:

- controlled tiling \(h=0\);
- practical halo \(h=64\) unless dataset/hardware protocol requires a registered change.

These are diagnostics, not substitutes for benchmark MAE/RMSE.

---

# 16. Statistical protocol

Pilot seed:

\[
42.
\]

Final claims:

\[
\boxed{\text{at least 3 seeds}}
\]

recommended:

\[
42,\ 123,\ 3407.
\]

LR sweep on validation only:

\[
\{10^{-4},3\times10^{-4},10^{-3}\}.
\]

Freeze LR before final matrix.

For B5 vs B3b, report:

\[
\Delta_i
=
|\hat N_i^{B5}-N_i|
-
|\hat N_i^{B3b}-N_i|
\]

and bootstrap a 95% CI of the mean paired difference.

Also report mean±std across seeds.

---

# 17. Kill rules

## K1

If:

\[
MAE(B5)\ge MAE(B2)
\]

across repeated seeds, inference-time reconciliation provides no added value beyond the regional head.

## K2

If:

\[
B3b\ge B5
\]

in accuracy at fair evidence scope, exact-adjoint superiority is unsupported.

## K3

If:

\[
B1\approx B5,
\]

regional training-only supervision is sufficient; inference-time claim fails.

## K4

If B5 only improves GAME/direct-tiled stability but not benchmark MAE/RMSE, it is not yet a CVPR-level counting result.

## K5

If measured latency/memory destroys the ultra-light efficiency story, weaken or abandon the efficiency claim.

## K6

If the carrier itself cannot approach the lightweight competitive range, do not interpret a tiny RMR gain as a strong paper result.

---

# 18. Required artifacts from every run

Each run directory must contain:

```text
resolved_config.yaml
train_log.csv
best_val_mae.pt
last.pt
profile.json
eval/
  predictions.csv
  summary.json
```

Mechanism runs additionally:

```text
eval_oracle/
  predictions.csv
  summary.json

eval_shuffled/
  predictions.csv
  summary.json
```

Final aggregation:

```text
paper_results/
  rq_matrix.csv
  seed_summary.csv
  paired_b5_vs_b3b.csv
  paired_b5_vs_b3b_bootstrap.json
  efficiency.csv
  region_scale_diagnostics.csv
  mechanism_iteration_diagnostics.csv
```

Never overwrite raw per-run outputs.

---

# 19. Expected repository tree

```text
rmr_count_agent_v2/
├── requirements.txt
├── run_matrix.sh
├── configs/
│   ├── direct.yaml
│   ├── region_loss.yaml
│   ├── region_aux.yaml
│   ├── local_refine.yaml
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

---

# 20. Mandatory unit tests

1. exact prefix sum;
2. exact rectangle queries;
3. exact adjoint identity;
4. positivity of every model output;
5. global count conservation;
6. zero regional residual produces zero exact correction direction;
7. solver strength 0 reduces iterative model to \(Y^{(0)}\);
8. bounded initial \(\eta=0.05\);
9. local-refine variant runs and stays positive;
10. learned-regional-project variant receives all region memberships and runs;
11. point rasterization preserves exact count.

The embedded reference project currently passes **11 tests**.

---

# 21. Canonical commands

Install:

```bash
cd rmr_count_agent_v2
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
pytest -q tests
```

Prepare manifests according to the dataset-specific code below.

Pilot LR sweep:

```bash
for lr in 1e-4 3e-4 1e-3; do
  python -m rmr_count.train \
    --config configs/rmr_t2.yaml \
    --seed 42 \
    --lr "$lr" \
    --output-dir "runs/sha_a/lr_sweep_rmr_t2_${lr}"
done
```

After LR selection on validation only:

```bash
LR=3e-4  # replace with validation-selected LR

for seed in 42 123 3407; do
  for cfg in direct region_loss region_aux local_refine learned_project rmr_t1 rmr_t2; do
    python -m rmr_count.train \
      --config "configs/${cfg}.yaml" \
      --seed "$seed" \
      --lr "$LR" \
      --output-dir "runs/sha_a/${cfg}_seed${seed}"
  done
done
```

Normal evaluation:

```bash
python -m rmr_count.eval \
  --checkpoint runs/sha_a/rmr_t2_seed42/best_val_mae.pt \
  --manifest data/sha_a_test.jsonl \
  --out-dir runs/sha_a/rmr_t2_seed42/eval \
  --region-mode predicted
```

Oracle mechanism diagnostic:

```bash
python -m rmr_count.eval \
  --checkpoint runs/sha_a/rmr_t2_seed42/best_val_mae.pt \
  --manifest data/sha_a_val.jsonl \
  --out-dir runs/sha_a/rmr_t2_seed42/eval_oracle \
  --region-mode oracle
```

Shuffled evidence diagnostic:

```bash
python -m rmr_count.eval \
  --checkpoint runs/sha_a/rmr_t2_seed42/best_val_mae.pt \
  --manifest data/sha_a_val.jsonl \
  --out-dir runs/sha_a/rmr_t2_seed42/eval_shuffled \
  --region-mode shuffled
```

Profile:

```bash
python -m rmr_count.profile \
  --variant rmr \
  --iterations 2 \
  --height 512 \
  --width 512 \
  --device cuda
```

---

# 22. Reference code

The following source files are the complete reference implementation. Agents may optimize kernels only after reproducing the registered behavior and tests. Semantic changes require a new experiment ID.



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
  for cfg in direct region_loss region_aux local_refine learned_project rmr_t1 rmr_t2; do
    python -m rmr_count.train \
      --config "configs/${cfg}.yaml" \
      --seed "$seed" \
      --lr "$LR" \
      --output-dir "runs/sha_a/${cfg}_seed${seed}"
  done
done
```


## File: `rmr_count/__init__.py`

```python

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
from typing import Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    RegionSet,
    build_multiscale_regions,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)

Variant = Literal[
    "direct",
    "region_loss",
    "region_aux",
    "local_refine",
    "learned_project",
    "rmr",
]


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        pad = k // 2
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=False),
            _gn(cout),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class TinyIR(nn.Module):
    """Small inverted residual block. Enabling component, not a novelty claim."""

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
    """Native local-first encoder exposing stride-4/8/16 features."""

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
    """Shared regional count regressor over exact integral-feature pooled descriptors."""

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
        b, _, h, w = f.shape
        pooled = region_average_features(f, regions.boxes)  # [B,M,C]
        geom = region_geometry(regions.boxes, h, w).to(dtype=f.dtype)
        geom = geom.unsqueeze(0).expand(b, -1, -1)
        raw = self.mlp(torch.cat([pooled, geom], dim=-1)).squeeze(-1)
        return F.softplus(raw).unsqueeze(1)  # [B,1,M]


class LocalPreconditioner(nn.Module):
    """Small bounded local preconditioner applied after the exact adjoint field."""

    def __init__(
        self,
        feature_dim: int = 32,
        hidden: int = 32,
        m_min: float = 0.25,
        m_max: float = 1.75,
    ):
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


class LocalCNNRefiner(nn.Module):
    """B3a control: purely local learned refinement.

    It has no regional count input. Spatial scope is deliberately limited to local 3x3 mixing.
    This asks whether simply spending extra local neural capacity can explain RMR's gain.
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.in_proj = ConvGNAct(feature_dim + 1, hidden, 1)
        self.dw = ConvGNAct(hidden, hidden, 3, groups=hidden)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.out(self.dw(self.in_proj(torch.cat([f, y], dim=1))))


class LearnedMembershipProjector(nn.Module):
    """B3b control: same regional residuals and same region memberships, learned allocation.

    For each rectangle R, the raw regional count residual delta_R = (A Y - b)_R is
    redistributed over cells p in R using a learned region-normalized visual weighting:

        pi_{R,p} = softmax_{p in R}(s_theta(F)_p)
        r_p = mean_{R contains p} delta_R * pi_{R,p}

    Exact RMR corresponds to a fixed uniform allocation delta_R / |R| before overlap
    averaging, followed by a small local preconditioner. This B3b control therefore has
    access to the same regional information and scope but is allowed to learn the
    region-to-grid allocation geometry.

    The explicit region loop is intentionally used for correctness in the causal control.
    Its measured latency must be reported; it is not proposed as the deployment model.
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            ConvGNAct(feature_dim + 1, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        self.post = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def project(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        raw_delta: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        b, _, h, w = y.shape
        score = self.score(torch.cat([f, y], dim=1))
        out = y.new_zeros((b, 1, h, w))
        coverage = y.new_zeros((b, 1, h, w))

        # Strong, fair control: each residual can influence every cell in its own region.
        for m, box in enumerate(regions.boxes.tolist()):
            y1, x1, y2, x2 = map(int, box)
            logits = score[:, :, y1:y2, x1:x2]
            flat = logits.flatten(-2)
            pi = torch.softmax(flat, dim=-1).view_as(logits)
            delta = raw_delta[:, :, m].view(b, 1, 1, 1)
            out[:, :, y1:y2, x1:x2] += delta * pi
            coverage[:, :, y1:y2, x1:x2] += 1.0

        field = out / coverage.clamp_min(1.0)
        # The post-net is allowed to shape the learned projection further.
        return self.post(torch.cat([f, y, field], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    # Pilot starts at 32 px; 16 px is added only if oracle/predicted-scale diagnostics justify it.
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = True
    iterations: int = 2

    # Stability: bounded step size with small initialization.
    eta_max: float = 0.20
    eta_init: float = 0.05
    residual_clip: float = 5.0

    eps: float = 1e-6


class RMRCount(nn.Module):
    """Regional Measure Reconciliation crowd counter and registered controls."""

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
        self.local_refiner = LocalCNNRefiner(cfg.feature_width) if variant == "local_refine" else None
        self.learned_projector = (
            LearnedMembershipProjector(cfg.feature_width) if variant == "learned_project" else None
        )

        n_steps = max(1, cfg.iterations)
        frac = cfg.eta_init / max(cfg.eta_max, 1e-8)
        init = _logit(frac)
        self.eta_logits = nn.Parameter(torch.full((n_steps,), init))

        # Training script ramps this from 0 -> 1 after the direct prediction has stabilized.
        self.solver_strength: float = 1.0

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

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

    def _raw_region_delta(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        return regional_sum(y, regions.boxes) - b_region

    def _rmr_field(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        """Coverage-normalized exact adjoint of regional count residual density."""
        _, _, h, w = y.shape
        delta = self._raw_region_delta(y, b_region, regions)  # [B,1,M]
        area = regions.area.to(y.dtype).view(1, 1, -1)
        residual_density = delta / area.clamp_min(1.0)

        back = regional_adjoint(residual_density, regions.boxes, h, w)
        coverage = regional_adjoint(
            torch.ones_like(residual_density), regions.boxes, h, w
        )
        r = back / coverage.clamp_min(1.0)
        if self.cfg.residual_clip > 0:
            r = r.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
        return r

    def forward(
        self,
        x: torch.Tensor,
        *,
        b_region_override: torch.Tensor | None = None,
        shuffle_region: bool = False,
    ) -> dict:
        c4, c8, c16 = self.encoder(x)
        f = self.fusion(c4, c8, c16)
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]
        regions = self._regions(h, w, x.device)

        out: dict = {
            "features": f,
            "z0": z0,
            "y0": y0,
            "regions": regions,
        }

        if self.region_head is not None:
            b_region = self.region_head(f, regions)
            if b_region_override is not None:
                if b_region_override.shape != b_region.shape:
                    raise ValueError(
                        f"b_region_override shape {tuple(b_region_override.shape)} "
                        f"!= {tuple(b_region.shape)}"
                    )
                b_region = b_region_override
            elif shuffle_region:
                # Shuffle only within each scale family to avoid a trivial scale mismatch artifact.
                pieces = []
                b_region = b_region.clone()
                for sid in torch.unique(regions.scale_id):
                    mask = regions.scale_id == sid
                    idx = torch.where(mask)[0]
                    if idx.numel() > 1:
                        perm = idx[torch.randperm(idx.numel(), device=idx.device)]
                        b_region[..., idx] = b_region[..., perm]
            out["b_region"] = b_region
        else:
            b_region = None

        if self.variant in {"direct", "region_loss", "region_aux"}:
            out["y"] = y0
            out["z"] = z0
            out["iterates"] = [y0]
            out["residual_fields"] = []
            return out

        z = z0
        y = y0
        iterates = [y0]
        residual_fields: list[torch.Tensor] = []

        if self.variant in {"learned_project", "rmr"} and b_region is None:
            raise RuntimeError(f"variant {self.variant} requires regional evidence")

        for t in range(self.cfg.iterations):
            eta = self._eta(t) * self.solver_strength
            if self.variant == "rmr":
                assert b_region is not None and self.preconditioner is not None
                r = self._rmr_field(y, b_region, regions)
                residual_fields.append(r)
                m = self.preconditioner(f, y, r)
                z = z - eta * m * torch.sigmoid(z) * r

            elif self.variant == "learned_project":
                assert b_region is not None and self.learned_projector is not None
                delta = self._raw_region_delta(y, b_region, regions)
                learned_field = self.learned_projector.project(f, y, delta, regions)
                if self.cfg.residual_clip > 0:
                    learned_field = learned_field.clamp(
                        -self.cfg.residual_clip, self.cfg.residual_clip
                    )
                residual_fields.append(learned_field)
                z = z - eta * learned_field

            elif self.variant == "local_refine":
                assert self.local_refiner is not None
                dz = self.local_refiner(f, y)
                if self.cfg.residual_clip > 0:
                    dz = dz.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
                residual_fields.append(dz)
                z = z - eta * dz

            else:
                raise RuntimeError(f"Unknown variant {self.variant}")

            y = F.softplus(z)
            iterates.append(y)

        out["y"] = y
        out["z"] = z
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
      local_refine:    direct + purely local learned inference refinement
      learned_project: region_aux + learned regional-membership projector
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
    if variant in {"local_refine", "learned_project", "rmr"} and len(iterates) > 2:
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
        eta_max=cfg["model"].get("eta_max", 0.20),
        eta_init=cfg["model"].get("eta_init", 0.05),
        residual_clip=cfg["model"].get("residual_clip", 5.0),
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
    solver_warmup_epochs = int(cfg["train"].get("solver_warmup_epochs", 5))
    solver_ramp_epochs = int(cfg["train"].get("solver_ramp_epochs", 20))

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
    fieldnames = ["epoch", "lr", "solver_strength", "eta0", "train_total", "train_cell", "train_global", "clip_rate", "residual_abs_mean", "residual_abs_max", "z_lt_minus10_frac", "val_MAE", "val_RMSE", "val_NAE", "val_Bias"]
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, epochs):
        model.train()

        # Stabilization protocol:
        # - first solver_warmup_epochs: direct prediction/regional head learn while solver update is off
        # - next solver_ramp_epochs: linearly ramp reconciliation/refinement strength to 1
        if model.variant in {"local_refine", "learned_project", "rmr"}:
            if epoch < solver_warmup_epochs:
                solver_strength = 0.0
            else:
                solver_strength = min(
                    1.0,
                    (epoch - solver_warmup_epochs + 1) / max(1, solver_ramp_epochs),
                )
            model.set_solver_strength(solver_strength)
        else:
            solver_strength = 0.0

        sums = {"total": 0.0, "cell": 0.0, "global": 0.0}
        n_steps = 0
        clipped = 0
        residual_abs_sum = 0.0
        residual_abs_max = 0.0
        z_sat_sum = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target_y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(image)
                losses = compute_losses(outputs, target, model.variant, loss_cfg)

            residuals = outputs.get("residual_fields", [])
            if residuals:
                r_last = residuals[-1].detach()
                residual_abs_sum += float(r_last.abs().mean().item())
                residual_abs_max = max(residual_abs_max, float(r_last.abs().max().item()))
            z_last = outputs.get("z")
            if z_last is not None:
                z_sat_sum += float((z_last.detach() < -10.0).float().mean().item())

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
            "solver_strength": solver_strength,
            "eta0": float(model._eta(0).detach().cpu().item()) if hasattr(model, "_eta") else 0.0,
            "train_total": sums["total"] / max(1, n_steps),
            "train_cell": sums["cell"] / max(1, n_steps),
            "train_global": sums["global"] / max(1, n_steps),
            "clip_rate": clipped / max(1, n_steps),
            "residual_abs_mean": residual_abs_sum / max(1, n_steps),
            "residual_abs_max": residual_abs_max,
            "z_lt_minus10_frac": z_sat_sum / max(1, n_steps),
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
            print(f"ep={epoch:04d} loss={row['train_total']:.4f} valMAE={metrics['MAE']:.3f} valRMSE={metrics['RMSE']:.3f} clip={row['clip_rate']:.3f} solver={solver_strength:.2f} rmax={row['residual_abs_max']:.3f}")
        else:
            print(f"ep={epoch:04d} loss={row['train_total']:.4f} clip={row['clip_rate']:.3f} solver={solver_strength:.2f} rmax={row['residual_abs_max']:.3f}")
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
        eta_max=cfg["model"].get("eta_max", 0.20),
        eta_init=cfg["model"].get("eta_init", 0.05),
        residual_clip=cfg["model"].get("residual_clip", 5.0),
    )
    model = RMRCount(mcfg, variant=cfg["model"]["variant"])
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_direct(
    model: RMRCount,
    image: torch.Tensor,
    target: torch.Tensor | None = None,
    region_mode: str = "predicted",
) -> tuple[torch.Tensor, dict]:
    """Direct prediction with optional mechanism diagnostics.

    region_mode:
      predicted : normal model
      oracle    : replace regional head output by exact GT regional counts (analysis only)
      shuffled  : shuffle predicted regional evidence within each scale family
    """
    x = image.unsqueeze(0)
    if region_mode == "predicted" or model.region_head is None:
        out = model(x)
    elif region_mode == "shuffled":
        out = model(x, shuffle_region=True)
    elif region_mode == "oracle":
        if target is None:
            raise ValueError("oracle region_mode requires target")
        # First pass is only to obtain the exact deterministic RegionSet.
        probe = model(x)
        regions = probe["regions"]
        b_gt = regional_sum(target.unsqueeze(0), regions.boxes)
        out = model(x, b_region_override=b_gt)
    else:
        raise ValueError(f"unknown region_mode={region_mode}")
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
    region_mode: str = "predicted",
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    region_errors: dict[object, list[float]] = defaultdict(list)
    mechanism_errors: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    region_count_bins: dict[str, list[float]] = defaultdict(list)

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].to(device)
            target = sample["target_y"].to(device)
            y, out = predict_direct(model, image, target=target, region_mode=region_mode)
            if region_mode == "predicted":
                y_t0 = predict_tiled(model, image, tile_size=tile_size, halo=0)
                y_th = predict_tiled(model, image, tile_size=tile_size, halo=practical_halo)
            else:
                # Oracle/shuffled modes are mechanism diagnostics, not deployment metrics.
                y_t0 = y
                y_th = y

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
            nmae = ae / t_reg.clamp_min(1.0)
            for sid in torch.unique(regions.scale_id):
                m = regions.scale_id == sid
                sid_i = int(sid.item())
                region_errors[sid_i].extend(ae[m].detach().cpu().tolist())
                region_errors[(sid_i, "nmae")].extend(
                    nmae[m].detach().cpu().tolist()
                )

            # Count-stratified regional diagnostics. These are more meaningful than
            # comparing a small-region MAE directly to whole-image MAE.
            gt_flat = t_reg.detach()
            ae_flat = ae.detach()
            bins = {
                "0": gt_flat == 0,
                "1": gt_flat == 1,
                "2_4": (gt_flat >= 2) & (gt_flat <= 4),
                "5_9": (gt_flat >= 5) & (gt_flat <= 9),
                "10p": gt_flat >= 10,
            }
            for name, mask in bins.items():
                if mask.any():
                    region_count_bins[name].extend(ae_flat[mask].cpu().tolist())

            # Mechanism trajectory: error to GT and disagreement with predicted b_R
            # at every iterate. This directly tests whether reconciliation reduces
            # regional inconsistency rather than merely changing the final count.
            iterates = out.get("iterates", [y])
            b_pred = out.get("b_region")
            for ti, yi in enumerate(iterates):
                q_i = regional_sum(yi.unsqueeze(0) if yi.ndim == 3 else yi, regions.boxes)[0, 0]
                gt_ae_i = (q_i - t_reg).abs()
                for sid in torch.unique(regions.scale_id):
                    m = regions.scale_id == sid
                    sid_i = int(sid.item())
                    mechanism_errors[(ti, sid_i, "gt_mae")].extend(
                        gt_ae_i[m].detach().cpu().tolist()
                    )
                if b_pred is not None:
                    b_i = b_pred[0, 0]
                    pred_dis_i = (q_i - b_i).abs()
                    for sid in torch.unique(regions.scale_id):
                        m = regions.scale_id == sid
                        sid_i = int(sid.item())
                        mechanism_errors[(ti, sid_i, "pred_disagreement")].extend(
                            pred_dis_i[m].detach().cpu().tolist()
                        )

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

    for sid_key, vals in region_errors.items():
        if isinstance(sid_key, tuple):
            sid, kind = sid_key
            name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
            summary[f"RegionNMAE_px_{name}"] = float(np.mean(vals)) if vals else float("nan")
        else:
            sid = sid_key
            name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
            summary[f"RegionMAE_px_{name}"] = float(np.mean(vals)) if vals else float("nan")
    for name, vals in region_count_bins.items():
        summary[f"RegionMAE_countbin_{name}"] = float(np.mean(vals)) if vals else float("nan")

    for (ti, sid, kind), vals in mechanism_errors.items():
        scale_name = "full" if sid == -1 else str(model.cfg.region_sizes_px[sid])
        summary[f"Iter{ti}_{kind}_px_{scale_name}"] = (
            float(np.mean(vals)) if vals else float("nan")
        )

    summary["region_mode"] = region_mode
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--practical-halo", type=int, default=64)
    ap.add_argument("--region-mode", choices=["predicted", "oracle", "shuffled"], default="predicted")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = make_model_from_ckpt(ckpt, device)
    ds = CrowdManifestDataset(args.manifest, train=False, output_stride=model.cfg.output_stride)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_eval)

    rows, summary = evaluate(
        model, loader, device, args.tile_size, args.practical_halo, region_mode=args.region_mode
    )
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
    ap.add_argument("--variant", default="rmr", choices=["direct", "region_loss", "region_aux", "local_refine", "learned_project", "rmr"])
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RMRCount(RMRConfig(iterations=args.iterations, region_sizes_px=(32,64,128), eta_max=0.20, eta_init=0.05, residual_clip=5.0), variant=args.variant).to(device).eval()
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


def test_local_refine_positive():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="local_refine")
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.all(out["y"] >= 0)
    assert len(out["residual_fields"]) == 2


def test_learned_project_same_regional_scope_runs():
    torch.manual_seed(0)
    model = RMRCount(
        RMRConfig(iterations=1, region_sizes_px=(32, 64)),
        variant="learned_project",
    )
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out["y"].shape == (1, 1, 32, 32)
    assert torch.all(out["y"] >= 0)
    assert out["b_region"].shape[-1] == out["regions"].boxes.shape[0]


def test_small_bounded_eta_initialization():
    model = RMRCount(RMRConfig(iterations=2, eta_max=0.2, eta_init=0.05), variant="rmr")
    eta0 = float(model._eta(0).detach())
    assert abs(eta0 - 0.05) < 1e-5


def test_solver_strength_zero_reduces_to_initial_measure():
    torch.manual_seed(0)
    model = RMRCount(RMRConfig(iterations=2), variant="rmr")
    model.set_solver_strength(0.0)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert torch.allclose(out["y"], out["y0"], atol=1e-7)
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


## File: `configs/direct.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/direct_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: direct
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/region_loss.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/region_loss_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: region_loss
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/region_aux.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/region_aux_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: region_aux
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/local_refine.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/local_refine_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: local_refine
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/learned_project.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/learned_project_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: learned_project
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/rmr_t1.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/rmr_t1_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: rmr
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 1
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


## File: `configs/rmr_t2.yaml`

```yaml
seed: 42
output_dir: runs/sha_a/rmr_t2_seed42
data:
  train_manifest: data/sha_a_train.jsonl
  val_manifest: data/sha_a_val.jsonl
  crop_size: 512
  scale_range:
  - 0.75
  - 1.25
model:
  variant: rmr
  output_stride: 4
  feature_width: 32
  region_sizes_px:
  - 32
  - 64
  - 128
  region_overlap: 0.5
  include_full_image: true
  iterations: 2
  eta_max: 0.2
  eta_init: 0.05
  residual_clip: 5.0
loss:
  lambda_global: 0.1
  lambda_region_map: 0.2
  lambda_region_head: 0.2
  lambda_deep_supervision: 0.1
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
  solver_warmup_epochs: 5
  solver_ramp_epochs: 20
```


# 23. Execution order for the agent

Run in this order only:

1. `pytest -q tests`.
2. Dataset manifest + count-conservation audit.
3. B0 carrier pilot.
4. LR selection on validation.
5. B1/B2.
6. B3a/B3b.
7. B4/B5.
8. predicted/oracle/shuffled mechanism diagnostics.
9. region-scale ablation.
10. multi-seed final matrix.
11. efficiency profiling on the same hardware/software stack.
12. only then write final paper tables/claims.

Do not skip directly to a large benchmark sweep before B3b vs B5 is known.

---

# 24. Agent completion criteria

The implementation agent is complete only when it produces:

- passing unit tests;
- reproducible configs;
- all B0–B5 runs or a documented kill decision;
- validation-selected LR provenance;
- per-image predictions;
- seed aggregation;
- paired B5-vs-B3b bootstrap CI;
- predicted/oracle/shuffled regional diagnostics;
- scale-wise regional NMAE/count-bin tables;
- iteration-wise regional-error trajectory;
- Params/FLOPs/latency/p95/memory;
- exact commit/hash or archive of code used for every table.

If a run fails, record the failure and stack trace. Do not silently alter the registered method.

---

# 25. Final scientific interpretation template

The agent must report one of these outcomes:

### Outcome A — core survives

B5 beats B2, B3a, and B3b with meaningful accuracy gains and competitive efficiency.

Then the paper can support:
> inference-time regional measure reconciliation contributes beyond regional supervision and generic/learned projection controls.

### Outcome B — regional inference helps, exact geometry does not

B5 beats B2 but B3b matches/beats B5.

Then:
> regional inference is useful, but exact-adjoint superiority is unsupported.

Do not claim operator geometry as the contribution.

### Outcome C — regional head is bottleneck

Oracle \(b_R\) strongly improves B5 while predicted \(b_R\) does not.

Then:
> reconciliation has headroom, but the current regional visual estimator is insufficient.

Improve only the regional estimator under a registered follow-up experiment; do not alter the central operator simultaneously.

### Outcome D — inference claim fails

B1/B2 match or beat B5.

Then the RQ is answered negatively. Stop/reframe.

### Outcome E — local learned capacity is enough

B3a matches/beats B5 at lower/equal real compute.

Then the ultra-light operator-efficiency argument is weak.

---

# 26. Bottom line

The implementation exists to test one question cleanly:

\[
\boxed{
\text{Does known regional count geometry add value at inference
that cannot be explained by supervision, auxiliary prediction,
or learned refinement under a lightweight budget?}
}
\]

Everything in the code and evaluation must serve that question.
