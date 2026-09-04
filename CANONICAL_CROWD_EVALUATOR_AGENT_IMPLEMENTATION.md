# Canonical Crowd-Counting Evaluator
## Agent Implementation Specification for FH-CMICF / PS-FH-CMICF

**Repository:** `minhphuc477/crowd-counting-lightweight`  
**Branch:** `MICF`  
**Primary files currently involved:**
- `hpc/metrics/counting.py`
- `tools/eval_micf_comprehensive.py`
- `hpc/losses/micf.py`
- `hpc/models/micf_lite.py`

**Purpose of this document:** replace the current mixed benchmark/debug evaluator with a scientifically defensible evaluator that clearly separates:

1. **cross-paper benchmark metrics**;
2. **PS-FH-CMICF method-validity metrics**;
3. **inference-decomposition diagnostics**;
4. **optional spatial diagnostics**;
5. **internal debugging metrics**.

This document is an implementation instruction for coding agents. Do not silently reinterpret formulas, rename metrics, or add new paper-facing metrics.

---

# 1. Non-negotiable evaluation policy

The evaluator must expose two clearly different layers.

## 1.1 Public benchmark layer

Use for comparison with crowd-counting papers:

\[
\boxed{\mathrm{MAE},\quad \mathrm{RMSE},\quad \mathrm{NAE}}
\]

where MAE and RMSE are the core universal counting metrics and NAE is additionally useful for normalized error and is official on NWPU-Crowd.

For ShanghaiTech A/B, the main comparison table may use only MAE/RMSE, but the evaluator may still compute NAE.

The public benchmark count must be computed at **whole-image level**, not from random training crops.

The canonical paper-facing prediction is the model's **full-image direct inference prediction**.

Do not present crop MAE, Window-MAE, PMAE aliases, or local tile error as the headline SOTA result.

---

## 1.2 Method-validity layer

Use only to analyze cumulative-field geometry:

\[
\boxed{\mathrm{VR}_{\tau},\quad \mathrm{NVR}}
\]

where:

- \(\mathrm{VR}_{\tau}\) = 2-increasing violation rate;
- \(\mathrm{NVR}\) = Negative Variation Ratio of the recovered signed measure.

These are **not** cross-paper crowd-counting benchmark metrics.

Their scientific role is to test whether a predicted cumulative field represents a valid non-negative counting measure.

For the main PS-FH mechanism table, compare like-for-like cumulative models such as B8 FH-CMICF and PS-FH-CMICF. Do not use NVR as an accuracy score against architectures that enforce \(Y\ge 0\) by construction.

---

# 2. Mathematical objects

Let image \(i\) have ground-truth crowd count:

\[
N_i \in \mathbb{N}_0
\]

and predicted count:

\[
\hat N_i\in\mathbb{R}.
\]

For cumulative methods, let the predicted cumulative field be:

\[
\hat C \in \mathbb{R}^{H_o\times W_o}.
\]

The recovered discrete counting measure is the mixed first difference:

\[
\boxed{
\hat Y_{ij}
=
\Delta_{xy}\hat C_{ij}
=
\hat C_{ij}
-\hat C_{i-1,j}
-\hat C_{i,j-1}
+\hat C_{i-1,j-1}
}
\]

with zero boundary convention outside the top and left boundaries.

The inverse transform is the two-dimensional prefix sum:

\[
\boxed{
\hat C_{ij}
=
\sum_{a\le i}\sum_{b\le j}\hat Y_{ab}.
}
\]

Therefore, on a fixed discrete grid:

\[
\boxed{
\sum_{ij}\hat Y_{ij}
=
\hat C_{H_o-1,W_o-1}.
}
\]

This telescoping identity is a **sanity condition**, not a performance metric.

---

# 3. Cross-paper benchmark metrics

## 3.1 Mean Absolute Error

For \(M\) test images:

\[
\boxed{
\mathrm{MAE}
=
\frac{1}{M}
\sum_{i=1}^{M}
|\hat N_i-N_i|
}
\]

Interpretation: average absolute whole-image counting error.

Lower is better.

---

## 3.2 Root Mean Squared Error

\[
\boxed{
\mathrm{RMSE}
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
(\hat N_i-N_i)^2
}
}
\]

Lower is better.

Important naming rule:

- call this `rmse` in code;
- call this **RMSE** in our paper;
- if comparing with papers that historically label the rooted quantity as "MSE", state once that their reported "MSE" corresponds to the RMSE formula.

Do not implement unrooted mean squared error under the benchmark key `mse`.

---

## 3.3 Normalized Absolute Error

Canonical definition:

\[
\boxed{
\mathrm{NAE}
=
\frac{1}{|I_+|}
\sum_{i\in I_+}
\frac{|\hat N_i-N_i|}{N_i},
\qquad
I_+=\{i:N_i>0\}.
}
\]

Images with \(N_i=0\) are excluded.

This matches the official NWPU-Crowd treatment of negative images.

Do **not** use:

\[
\frac{|\hat N_i-N_i|}{\max(N_i,\epsilon)}
\]

for benchmark NAE because it makes zero-count images produce arbitrarily huge normalized error and does not match the NWPU official protocol.

If every image has \(N_i=0\), return `NaN`, not zero. This makes the undefined state explicit.

---

# 4. Dataset-specific paper reporting

## 4.1 ShanghaiTech Part A / Part B

Paper main table:

- MAE
- RMSE

Optional supplementary:

- NAE

---

## 4.2 UCF-QNRF

Report:

- MAE
- RMSE
- NAE

---

## 4.3 NWPU-Crowd

Report:

- Overall MAE
- Overall RMSE
- Overall NAE

When official scene/luminance labels are available, additionally report the benchmark's official category breakdown.

Do not invent alternative NWPU bins.

---

## 4.4 JHU-CROWD++

Main:

- MAE
- RMSE

If the evaluator has access to official category labels, supplementary category analysis may include the official low/medium/high/weather/distractor structure.

---

# 5. Cumulative-field validity

A valid non-negative bivariate counting measure requires non-negative rectangle increments. In the discrete cell representation this becomes:

\[
\boxed{
\Delta_{xy}\hat C_{ij}\ge 0.
}
\]

Thus invalidity is directly visible in the recovered measure:

\[
\hat Y=\Delta_{xy}\hat C.
\]

---

# 6. Violation Rate

Choose a fixed numerical tolerance:

\[
\tau=10^{-6}.
\]

For one image:

\[
\boxed{
\mathrm{VR}_{\tau}
=
\frac{
\#\{(i,j):\hat Y_{ij}<-\tau\}
}{
H_oW_o
}.
}
\]

Meaning:

> fraction of recovered cells that violate non-negativity beyond numerical tolerance.

The paper-facing dataset-level value is the **micro** rate:

\[
\boxed{
\mathrm{VR}^{\mathrm{micro}}_{\tau}
=
\frac{
\sum_n \#\{(i,j):\hat Y^{(n)}_{ij}<-\tau\}
}{
\sum_n H_o^{(n)}W_o^{(n)}
}.
}
\]

Also compute the macro image average for debugging:

\[
\mathrm{VR}^{\mathrm{macro}}_{\tau}
=
\frac1M
\sum_n
\mathrm{VR}^{(n)}_{\tau}.
\]

Policy:

- paper mechanism table: `vr_tau_micro`;
- supplementary/debugging: `vr_tau_macro`.

Do not publish raw-zero-threshold and tolerance-threshold duplicates as separate headline metrics.

---

# 7. Negative Variation Ratio

For recovered signed measure \(\hat Y\), define positive and negative variations:

\[
\hat Y^+_{ij}
=
\max(\hat Y_{ij},0)
\]

and

\[
\hat Y^-_{ij}
=
\max(-\hat Y_{ij},0).
\]

Positive variation mass:

\[
P
=
\sum_{ij}\hat Y^+_{ij}.
\]

Negative variation mass:

\[
Q
=
\sum_{ij}\hat Y^-_{ij}.
\]

Define:

\[
\boxed{
\mathrm{NVR}
=
\frac{Q}{\max(P,\epsilon)}.
}
\]

Use:

\[
\epsilon=10^{-12}
\]

in float64 aggregation.

This is intentionally **not**:

\[
\frac{Q}{\sum|\hat Y|}.
\]

The old denominator \(\sum|\hat Y|\) measures the negative share of total variation but does not equal negative-to-positive variation ratio.

For the dataset-level paper metric use pooled variation:

\[
\boxed{
\mathrm{NVR}^{\mathrm{micro}}
=
\frac{
\sum_n Q_n
}{
\max(\sum_n P_n,\epsilon)
}.
}
\]

Also compute:

\[
\mathrm{NVR}^{\mathrm{macro}}
=
\frac1M
\sum_n
\frac{Q_n}{\max(P_n,\epsilon)}
\]

for diagnostics.

Policy:

- paper mechanism table: `nvr_micro`;
- supplementary/debugging: `nvr_macro`.

Terminology:

- old internal name: `negative_mass_ratio`;
- new paper-facing name: **Negative Variation Ratio (NVR)**;
- do not claim negative variation itself causes count bias;
- say that it measures geometric inconsistency with a valid non-negative measure.

---

# 8. Direct-versus-tiled decomposition discrepancy

Direct and tiled predictions for the same image are:

\[
\hat N_i^{D},
\qquad
\hat N_i^{T}.
\]

Absolute prediction discrepancy:

\[
\boxed{
D_{\mathrm{abs}}
=
\frac1M
\sum_i
|\hat N_i^{D}-\hat N_i^{T}|.
}
\]

Normalized discrepancy:

\[
\boxed{
D_{\mathrm{norm}}
=
\frac1M
\sum_i
\frac{
|\hat N_i^{D}-\hat N_i^{T}|
}{
\max(N_i,1)
}.
}
\]

Use the descriptive paper name:

> normalized direct-tiled prediction discrepancy

Do not turn this into a fake universal SOTA metric.

Scientific role:

> quantify sensitivity of the same trained model to inference decomposition.

For PS-FH-CMICF, reduction in this quantity supports the claim that strict-local finite-horizon learned operations reduce full-image/tiled inconsistency.

---

# 9. GAME spatial metric

GAME at level \(L\) is:

\[
\boxed{
\mathrm{GAME}(L)
=
\frac1M
\sum_{i=1}^{M}
\sum_{r=1}^{4^L}
|\hat N_{ir}-N_{ir}|.
}
\]

Each image is divided into:

\[
2^L\times2^L
\]

non-overlapping regions.

Important identity:

\[
\boxed{
\mathrm{GAME}(0)=\mathrm{MAE}.
}
\]

For our cell-count measure output, the model does not predict a continuous pixel density. Therefore define a **piecewise-uniform pixel measure** only for GAME evaluation:

- cell \((i,j)\) has total mass \(\hat Y_{ij}\);
- its physical support is clipped to the actual image:
  \[
  [js,\min((j+1)s,W))
  \times
  [is,\min((i+1)s,H));
  \]
- distribute the cell mass uniformly over that actual support;
- integrate fractional overlap with each GAME rectangle.

This convention preserves each edge cell's full mass even when the image size is not divisible by stride.

Do not use the old implementation that divides overlap by the full \(s\times s\) support for clipped edge cells, because then GAME(0) can lose edge mass and fail to equal MAE.

GAME is optional/supplementary for this paper.

Do not publish `GAME@stride16` as a benchmark metric. It is an internal grid diagnostic.

---

# 10. Metrics that are not canonical paper metrics

The following may remain in raw debug files but must not appear in the canonical benchmark summary:

- SRE;
- signed bias;
- median AE;
- P90 AE;
- P95 AE;
- max AE;
- Window-MAE;
- Window-RMSE;
- PMAE aliases;
- PRMSE aliases;
- full-window/edge-window custom metrics;
- cancellation ratio;
- cumulative-field NMAE;
- local-measure NL1;
- violation magnitude;
- total negative mass as a headline score;
- stride-space GAME.

Conservation error is not a score. Treat it as a numerical assertion:

\[
\left|
\hat C[-1,-1]-\sum\Delta_{xy}\hat C
\right|
<\varepsilon_{\mathrm{cons}}.
\]

Recommended:

\[
\varepsilon_{\mathrm{cons}}=10^{-4}
\]

in FP32 evaluation, or a relative version when counts are very large.

---

# 11. Required output schema

The final `summary.json` must be nested so benchmark and mechanism metrics cannot be confused.

Example:

```json
{
  "benchmark": {
    "direct": {
      "mae": 0.0,
      "rmse": 0.0,
      "nae": 0.0,
      "num_images": 182
    },
    "tiled_controlled": {
      "mae": 0.0,
      "rmse": 0.0,
      "nae": 0.0
    },
    "tiled_practical": {
      "mae": 0.0,
      "rmse": 0.0,
      "nae": 0.0
    }
  },
  "method_validity": {
    "direct": {
      "vr_tau_micro": 0.0,
      "vr_tau_macro": 0.0,
      "nvr_micro": 0.0,
      "nvr_macro": 0.0,
      "positive_variation_total": 0.0,
      "negative_variation_total": 0.0
    }
  },
  "decomposition": {
    "controlled": {
      "mean_abs_prediction_discrepancy": 0.0,
      "mean_normalized_prediction_discrepancy": 0.0
    },
    "practical": {
      "mean_abs_prediction_discrepancy": 0.0,
      "mean_normalized_prediction_discrepancy": 0.0
    }
  },
  "spatial_optional": {
    "direct": {
      "game_0": 0.0,
      "game_1": 0.0,
      "game_2": 0.0,
      "game_3": 0.0
    }
  },
  "sanity": {
    "max_conservation_error": 0.0
  }
}
```

Do not flatten private diagnostics into the benchmark namespace.

---

# 12. Implementation: `hpc/metrics/counting.py`

Replace/extend the metric implementation with the following code.

```python
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_flat(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _validate_pair(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    preds = _as_flat(predictions)
    gts = _as_flat(targets)

    if preds.shape != gts.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{preds.shape} vs {gts.shape}"
        )

    if not np.isfinite(preds).all():
        raise ValueError(
            "Predictions contain non-finite values (NaN or Inf)"
        )

    if not np.isfinite(gts).all():
        raise ValueError(
            "Ground-truth targets contain non-finite values (NaN or Inf)"
        )

    return preds, gts


def compute_mae(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    return float(
        np.mean(
            np.abs(preds - gts)
        )
    )


def compute_rmse(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    err = preds - gts

    return float(
        np.sqrt(
            np.mean(err * err)
        )
    )


def compute_nae(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    """
    Canonical crowd-counting NAE.

    Matches the NWPU-Crowd treatment:
    samples with gt == 0 are excluded.

    NAE =
        mean_{i: gt_i > 0}
        |pred_i - gt_i| / gt_i

    Returns NaN if there is no positive-ground-truth sample.
    """
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    mask = gts > 0.0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                preds[mask] - gts[mask]
            )
            / gts[mask]
        )
    )


def benchmark_count_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    """
    Paper-facing crowd-counting benchmark summary.

    Keep this function intentionally small.
    """
    preds, gts = _validate_pair(predictions, targets)

    return {
        "mae": compute_mae(preds, gts),
        "rmse": compute_rmse(preds, gts),
        "nae": compute_nae(preds, gts),
        "num_images": int(preds.size),
    }


# ------------------------------------------------------------------
# Diagnostics below are NOT part of the canonical benchmark summary.
# Keep them only if other code depends on them.
# ------------------------------------------------------------------

def compute_bias(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    return float(
        np.mean(preds - gts)
    )


def compute_sre(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    """
    Diagnostic only.

    SRE = sqrt(
        mean_{i: gt_i > 0}
        (pred_i - gt_i)^2 / gt_i
    )

    Zero-GT samples are excluded for the same reason as NAE.
    """
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    mask = gts > 0.0

    if not np.any(mask):
        return float("nan")

    err = preds[mask] - gts[mask]

    return float(
        np.sqrt(
            np.mean(
                (err * err)
                / gts[mask]
            )
        )
    )


def diagnostic_count_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return {
            "signed_bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "p95_ae": float("nan"),
            "max_ae": float("nan"),
            "sre": float("nan"),
        }

    err = preds - gts
    ae = np.abs(err)

    return {
        "signed_bias": float(np.mean(err)),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(np.max(ae)),
        "sre": compute_sre(preds, gts),
    }


def count_metric_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
    eps: float | None = None,
) -> Dict[str, float]:
    """
    Backward-compatible extended summary.

    New evaluator code should call benchmark_count_summary().
    `eps` is accepted only so old callers do not break.
    """
    del eps

    out = benchmark_count_summary(
        predictions,
        targets,
    )
    out.update(
        diagnostic_count_summary(
            predictions,
            targets,
        )
    )
    return out


def evaluate_counting_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    """
    Backward-compatible entry point.
    """
    out = benchmark_count_summary(
        predictions,
        targets,
    )
    out["bias"] = compute_bias(
        predictions,
        targets,
    )
    return out
```

---

# 13. New implementation: `hpc/metrics/micf_validity.py`

Create this file.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ValidityRow:
    vr_tau: float
    positive_variation: float
    negative_variation: float
    nvr: float
    violating_cells: int
    total_cells: int


def recovered_measure_validity(
    y_pred: torch.Tensor,
    *,
    tau: float = 1e-6,
    eps: float = 1e-12,
) -> ValidityRow:
    """
    Validity diagnostics for a recovered signed counting measure.

    Parameters
    ----------
    y_pred:
        Recovered measure Y = Delta_xy C.
        Any shape is allowed; all cells are pooled.
    tau:
        Numerical tolerance in count units.
    eps:
        Stabilizer for NVR denominator.

    Returns
    -------
    ValidityRow
        Per-sample quantities.
    """
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError(
            "y_pred must be a torch.Tensor"
        )

    y = (
        y_pred.detach()
        .to(dtype=torch.float64)
    )

    if not torch.isfinite(y).all():
        raise ValueError(
            "Recovered measure contains NaN/Inf"
        )

    positive = torch.clamp(
        y,
        min=0.0,
    )

    negative = torch.clamp(
        -y,
        min=0.0,
    )

    positive_variation = float(
        positive.sum().item()
    )

    negative_variation = float(
        negative.sum().item()
    )

    violating_cells = int(
        (y < -float(tau))
        .sum()
        .item()
    )

    total_cells = int(
        y.numel()
    )

    vr_tau = float(
        violating_cells
        / max(total_cells, 1)
    )

    nvr = float(
        negative_variation
        / max(
            positive_variation,
            float(eps),
        )
    )

    return ValidityRow(
        vr_tau=vr_tau,
        positive_variation=positive_variation,
        negative_variation=negative_variation,
        nvr=nvr,
        violating_cells=violating_cells,
        total_cells=total_cells,
    )


def aggregate_validity(
    rows: Iterable[ValidityRow],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    rows = list(rows)

    if not rows:
        return {
            "vr_tau_micro": float("nan"),
            "vr_tau_macro": float("nan"),
            "nvr_micro": float("nan"),
            "nvr_macro": float("nan"),
            "positive_variation_total": 0.0,
            "negative_variation_total": 0.0,
            "violating_cells_total": 0,
            "cells_total": 0,
        }

    violating_cells_total = int(
        sum(
            r.violating_cells
            for r in rows
        )
    )

    cells_total = int(
        sum(
            r.total_cells
            for r in rows
        )
    )

    positive_total = float(
        sum(
            r.positive_variation
            for r in rows
        )
    )

    negative_total = float(
        sum(
            r.negative_variation
            for r in rows
        )
    )

    vr_tau_micro = float(
        violating_cells_total
        / max(cells_total, 1)
    )

    vr_tau_macro = float(
        np.mean(
            [r.vr_tau for r in rows]
        )
    )

    nvr_micro = float(
        negative_total
        / max(
            positive_total,
            float(eps),
        )
    )

    nvr_macro = float(
        np.mean(
            [r.nvr for r in rows]
        )
    )

    return {
        "vr_tau_micro": vr_tau_micro,
        "vr_tau_macro": vr_tau_macro,
        "nvr_micro": nvr_micro,
        "nvr_macro": nvr_macro,
        "positive_variation_total": positive_total,
        "negative_variation_total": negative_total,
        "violating_cells_total": violating_cells_total,
        "cells_total": cells_total,
    }
```

---

# 14. New implementation: `hpc/metrics/decomposition.py`

Create this file.

```python
from __future__ import annotations

from typing import Union

import numpy as np
import torch


ArrayLike = Union[
    np.ndarray,
    torch.Tensor,
]


def _flat(
    x: ArrayLike,
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    arr = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    if not np.isfinite(arr).all():
        raise ValueError(
            "Non-finite values in decomposition metric input"
        )

    return arr


def direct_tiled_discrepancy(
    direct_predictions: ArrayLike,
    tiled_predictions: ArrayLike,
    ground_truth: ArrayLike,
) -> dict[str, float]:
    d = _flat(direct_predictions)
    t = _flat(tiled_predictions)
    g = _flat(ground_truth)

    if not (
        d.shape == t.shape == g.shape
    ):
        raise ValueError(
            f"Shape mismatch: "
            f"direct={d.shape}, "
            f"tiled={t.shape}, "
            f"gt={g.shape}"
        )

    if d.size == 0:
        return {
            "mean_abs_prediction_discrepancy": float("nan"),
            "mean_normalized_prediction_discrepancy": float("nan"),
        }

    delta = np.abs(d - t)

    return {
        "mean_abs_prediction_discrepancy":
            float(
                np.mean(delta)
            ),
        "mean_normalized_prediction_discrepancy":
            float(
                np.mean(
                    delta
                    / np.maximum(g, 1.0)
                )
            ),
    }
```

---

# 15. New implementation: `hpc/metrics/game.py`

Create this file.

The implementation below treats each predicted output cell as a count mass uniformly distributed over the cell's **actual in-image support**. This makes edge handling mass preserving.

```python
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _points_array(
    points: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(
        points,
        dtype=np.float64,
    )

    if arr.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    return arr.reshape(-1, 2)


def _cell_actual_bounds(
    row: int,
    col: int,
    *,
    stride: int,
    image_h: int,
    image_w: int,
) -> tuple[float, float, float, float]:
    y0 = float(
        row * stride
    )
    x0 = float(
        col * stride
    )

    y1 = float(
        min(
            (row + 1) * stride,
            image_h,
        )
    )

    x1 = float(
        min(
            (col + 1) * stride,
            image_w,
        )
    )

    return (
        x0,
        y0,
        x1,
        y1,
    )


def _intersection_area(
    a_x0: float,
    a_y0: float,
    a_x1: float,
    a_y1: float,
    b_x0: float,
    b_y0: float,
    b_x1: float,
    b_y1: float,
) -> float:
    w = max(
        0.0,
        min(a_x1, b_x1)
        - max(a_x0, b_x0),
    )

    h = max(
        0.0,
        min(a_y1, b_y1)
        - max(a_y0, b_y0),
    )

    return float(
        w * h
    )


def region_count_from_cell_measure(
    pred_y: torch.Tensor,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_h: int,
    image_w: int,
    stride: int,
) -> float:
    """
    Integrate a cell-count measure over a pixel-space rectangle.

    Each output cell's total mass is uniformly distributed
    over its actual in-image support.

    This is an evaluation convention used only for GAME.
    """
    if pred_y.ndim == 4:
        if (
            pred_y.shape[0] != 1
            or pred_y.shape[1] != 1
        ):
            raise ValueError(
                f"Expected [1,1,H,W], got "
                f"{tuple(pred_y.shape)}"
            )
        y = (
            pred_y[0, 0]
            .detach()
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    elif pred_y.ndim == 2:
        y = (
            pred_y.detach()
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    else:
        raise ValueError(
            f"Expected 2D or [1,1,H,W], got "
            f"{tuple(pred_y.shape)}"
        )

    out_h, out_w = y.shape

    total = 0.0

    for r in range(out_h):
        for c in range(out_w):
            (
                cx0,
                cy0,
                cx1,
                cy1,
            ) = _cell_actual_bounds(
                r,
                c,
                stride=stride,
                image_h=image_h,
                image_w=image_w,
            )

            support_area = (
                (cx1 - cx0)
                * (cy1 - cy0)
            )

            if support_area <= 0.0:
                continue

            overlap = _intersection_area(
                cx0,
                cy0,
                cx1,
                cy1,
                x0,
                y0,
                x1,
                y1,
            )

            if overlap <= 0.0:
                continue

            total += float(
                y[r, c]
                * (
                    overlap
                    / support_area
                )
            )

    return float(total)


def _gt_region_count(
    points: np.ndarray,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    if points.size == 0:
        return 0.0

    mask = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )

    return float(
        mask.sum()
    )


def game_errors_one_image(
    pred_y: torch.Tensor,
    points: np.ndarray,
    *,
    image_h: int,
    image_w: int,
    stride: int,
    levels: Iterable[int] = (0, 1, 2, 3),
) -> dict[int, float]:
    pts = _points_array(
        points
    )

    out: dict[int, float] = {}

    for level_raw in levels:
        level = int(
            level_raw
        )

        if level < 0:
            raise ValueError(
                "GAME level must be >= 0"
            )

        parts = 2 ** level

        x_edges = np.linspace(
            0.0,
            float(image_w),
            parts + 1,
        )

        y_edges = np.linspace(
            0.0,
            float(image_h),
            parts + 1,
        )

        total_abs_error = 0.0

        for r in range(parts):
            for c in range(parts):
                x0 = float(
                    x_edges[c]
                )
                x1 = float(
                    x_edges[c + 1]
                )
                y0 = float(
                    y_edges[r]
                )
                y1 = float(
                    y_edges[r + 1]
                )

                pred_count = (
                    region_count_from_cell_measure(
                        pred_y,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        image_h=image_h,
                        image_w=image_w,
                        stride=stride,
                    )
                )

                gt_count = (
                    _gt_region_count(
                        pts,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                )

                total_abs_error += abs(
                    pred_count
                    - gt_count
                )

        out[level] = float(
            total_abs_error
        )

    return out


def aggregate_game(
    per_image_game: list[dict[int, float]],
) -> dict[str, float]:
    if not per_image_game:
        return {}

    levels = sorted(
        set().union(
            *[
                set(x.keys())
                for x in per_image_game
            ]
        )
    )

    out: dict[str, float] = {}

    for level in levels:
        vals = [
            float(x[level])
            for x in per_image_game
            if level in x
        ]

        out[
            f"game_{level}"
        ] = float(
            np.mean(vals)
        )

    return out
```

---

# 16. Important GAME invariant

For every image:

\[
\sum_{\text{GAME regions at level }L}
\hat N_r
=
\sum_{ij}\hat Y_{ij}.
\]

Therefore, for level zero:

\[
\mathrm{GAME}_i(0)
=
|
\hat N_i-N_i
|.
\]

Across the test set:

\[
\boxed{
\mathrm{GAME}(0)=\mathrm{MAE}.
}
\]

The evaluator must assert this numerically.

Recommended tolerance:

```python
assert abs(
    game_summary["game_0"]
    - benchmark_summary["mae"]
) <= 1e-4
```

Use a slightly larger relative tolerance only if FP32 accumulation at extremely large counts requires it.

---

# 17. Patch `tools/eval_micf_comprehensive.py`

## 17.1 Replace imports

Add:

```python
from hpc.metrics.counting import (
    benchmark_count_summary,
    diagnostic_count_summary,
)

from hpc.metrics.micf_validity import (
    ValidityRow,
    aggregate_validity,
    recovered_measure_validity,
)

from hpc.metrics.decomposition import (
    direct_tiled_discrepancy,
)

from hpc.metrics.game import (
    aggregate_game,
    game_errors_one_image,
)
```

The existing import:

```python
from hpc.metrics.counting import count_metric_summary
```

should no longer be the canonical path for final benchmark output.

---

## 17.2 Delete/retire duplicate validity functions

The current evaluator contains local definitions resembling:

```python
def measure_validity_metrics(...)
def aggregate_validity_metrics(...)
```

Retire them from canonical use.

Reason:

- current NMR denominator is `sum(abs(Y))`;
- canonical NVR denominator must be positive variation;
- current code duplicates raw and thresholded VR;
- validity logic belongs in one unit-tested metric module.

If old CSV compatibility is needed, old keys may be written under:

```text
legacy_debug
```

but never under `method_validity`.

---

## 17.3 Direct inference must return valid field

Use the existing model behavior:

```python
pred_count_direct, pred_field_direct = model.predict(
    image,
    pad_multiple=args.direct_pad_multiple,
)
```

`MICFLite.predict()` already:

1. pads input as required;
2. performs forward pass;
3. crops the output field to:
   ```python
   out_h = math.ceil(h / output_stride)
   out_w = math.ceil(w / output_stride)
   ```
4. returns count from the valid field.

For cumulative heads:

```python
pred_c_direct = pred_field_direct
pred_y_direct = discrete_mixed_difference(
    pred_c_direct
)
```

For local heads:

```python
pred_y_direct = pred_field_direct
```

Do not use the padded full field for benchmark validity.

---

# 18. Per-image canonical row

For every test image write one row to:

```text
per_image_canonical.csv
```

Recommended columns:

```python
row = {
    "image_index": image_index,
    "image_name": image_name,
    "height": H,
    "width": W,
    "gt_count": gt_count,

    "direct_pred_count": direct_pred_count,
    "controlled_tiled_pred_count":
        controlled_tiled_pred_count,
    "practical_tiled_pred_count":
        practical_tiled_pred_count,

    "direct_abs_error":
        abs(direct_pred_count - gt_count),

    "controlled_tiled_abs_error":
        abs(
            controlled_tiled_pred_count
            - gt_count
        ),

    "practical_tiled_abs_error":
        abs(
            practical_tiled_pred_count
            - gt_count
        ),

    "direct_vs_controlled_abs":
        abs(
            direct_pred_count
            - controlled_tiled_pred_count
        ),

    "direct_vs_practical_abs":
        abs(
            direct_pred_count
            - practical_tiled_pred_count
        ),
}
```

For cumulative methods add:

```python
valid = recovered_measure_validity(
    pred_y_direct,
    tau=1e-6,
)

row.update(
    {
        "direct_vr_tau":
            valid.vr_tau,
        "direct_positive_variation":
            valid.positive_variation,
        "direct_negative_variation":
            valid.negative_variation,
        "direct_nvr":
            valid.nvr,
        "direct_violating_cells":
            valid.violating_cells,
        "direct_total_cells":
            valid.total_cells,
    }
)
```

---

# 19. Canonical benchmark aggregation

Collect arrays:

```python
gt_counts = np.asarray(
    gt_counts,
    dtype=np.float64,
)

direct_counts = np.asarray(
    direct_counts,
    dtype=np.float64,
)

controlled_counts = np.asarray(
    controlled_counts,
    dtype=np.float64,
)

practical_counts = np.asarray(
    practical_counts,
    dtype=np.float64,
)
```

Then:

```python
benchmark_direct = (
    benchmark_count_summary(
        direct_counts,
        gt_counts,
    )
)

benchmark_controlled = (
    benchmark_count_summary(
        controlled_counts,
        gt_counts,
    )
)

benchmark_practical = (
    benchmark_count_summary(
        practical_counts,
        gt_counts,
    )
)
```

---

# 20. Validity aggregation

Maintain:

```python
direct_validity_rows: list[ValidityRow] = []
controlled_validity_rows: list[ValidityRow] = []
practical_validity_rows: list[ValidityRow] = []
```

Then:

```python
validity_direct = (
    aggregate_validity(
        direct_validity_rows
    )
)

validity_controlled = (
    aggregate_validity(
        controlled_validity_rows
    )
)

validity_practical = (
    aggregate_validity(
        practical_validity_rows
    )
)
```

---

# 21. Decomposition aggregation

```python
controlled_decomposition = (
    direct_tiled_discrepancy(
        direct_counts,
        controlled_counts,
        gt_counts,
    )
)

practical_decomposition = (
    direct_tiled_discrepancy(
        direct_counts,
        practical_counts,
        gt_counts,
    )
)
```

Do not compute only:

```python
direct_mae - tiled_mae
```

because differences between aggregate errors are not the same as paired prediction discrepancy.

---

# 22. GAME aggregation

During image loop:

```python
direct_game_rows.append(
    game_errors_one_image(
        pred_y_direct,
        points_in_bounds,
        image_h=H,
        image_w=W,
        stride=model.output_stride,
        levels=args.game_levels,
    )
)
```

After loop:

```python
game_direct = (
    aggregate_game(
        direct_game_rows
    )
)
```

Then enforce:

```python
if 0 in args.game_levels:
    mae = float(
        benchmark_direct["mae"]
    )

    game0 = float(
        game_direct["game_0"]
    )

    tol = 1e-4 * max(
        1.0,
        abs(mae),
    )

    if abs(game0 - mae) > tol:
        raise RuntimeError(
            "GAME(0) invariant failed: "
            f"GAME0={game0:.8f}, "
            f"MAE={mae:.8f}, "
            f"tol={tol:.8f}"
        )
```

---

# 23. Conservation assertion

For cumulative output:

```python
pred_c = pred_field_direct.float()
pred_y = discrete_mixed_difference(
    pred_c
)

count_from_c = float(
    pred_c[..., -1, -1]
    .reshape(-1)[0]
    .item()
)

count_from_y = float(
    pred_y.sum().item()
)

abs_cons_error = abs(
    count_from_c
    - count_from_y
)

cons_tol = (
    1e-4
    * max(
        1.0,
        abs(count_from_c),
    )
)

if abs_cons_error > cons_tol:
    raise RuntimeError(
        "Cumulative/measure conservation failed: "
        f"C[-1,-1]={count_from_c:.8f}, "
        f"sum(Y)={count_from_y:.8f}, "
        f"error={abs_cons_error:.8f}, "
        f"tol={cons_tol:.8f}"
    )
```

Store maximum observed conservation error only for the `sanity` section.

Do not treat lower conservation error as a model-performance improvement when both methods already satisfy numerical precision.

---

# 24. Final summary assembly

```python
summary = {
    "benchmark": {
        "direct":
            benchmark_direct,
        "tiled_controlled":
            benchmark_controlled,
        "tiled_practical":
            benchmark_practical,
    },

    "method_validity": {
        "direct":
            validity_direct,
        "tiled_controlled":
            validity_controlled,
        "tiled_practical":
            validity_practical,
    },

    "decomposition": {
        "controlled":
            controlled_decomposition,
        "practical":
            practical_decomposition,
    },

    "spatial_optional": {
        "direct":
            game_direct,
    },

    "sanity": {
        "max_conservation_error":
            max_conservation_error,
    },

    "protocol": {
        "split":
            args.split,
        "tile_size":
            args.tile_size,
        "controlled_halo":
            args.controlled_halo,
        "practical_halo":
            args.halo,
        "direct_pad_multiple":
            args.direct_pad_multiple,
        "game_levels":
            list(args.game_levels),
        "validity_tau":
            1e-6,
        "nvr_eps":
            1e-12,
        "benchmark_primary_inference":
            "full_image_direct",
    },
}
```

Write with:

```python
with (
    output_dir
    / "summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=True,
    )
```

---

# 25. Console output

The console should be short and distinguish paper metrics from diagnostics.

Recommended:

```text
================================================================================
CANONICAL CROWD-COUNTING EVALUATION
================================================================================

Benchmark / Full-Image Direct
MAE   : ...
RMSE  : ...
NAE   : ...

PS-FH Cumulative Validity / Direct
VR_tau micro : ... %
NVR micro    : ... %

Inference Decomposition
Direct vs Controlled | normalized discrepancy : ...
Direct vs Practical  | normalized discrepancy : ...

Optional Spatial
GAME(0): ...
GAME(1): ...
GAME(2): ...
GAME(3): ...

Sanity
max |C[-1,-1] - sum(Delta_xy C)| : ...
================================================================================
```

Do not print 30 metrics in the main console table.

---

# 26. Legacy window metrics

Current evaluator contains window-level code and aliases such as:

- `window_mae`;
- `window_rmse`;
- `pmae`;
- `prmse`;
- `pmae_micro`;
- `pmae_macro`;
- edge/full-window versions.

Policy:

1. Keep them only if needed for historical B5b/B8 analysis.
2. Move their JSON output under:
   ```text
   legacy_debug.window
   ```
3. Remove PMAE/PRMSE aliases from paper-facing output.
4. Never call our fixed 256 tile diagnostic PaDNet PMAE unless we exactly reproduce the PaDNet patch protocol.

Preferred internal naming:

```text
tile256_debug_mae
tile256_debug_rmse
```

not PMAE/PRMSE.

---

# 27. Representation diagnostics

These current metrics are useful while debugging but are not universal benchmark quantities:

\[
\text{Cumulative Field NMAE}
\]

and:

\[
\text{Measure NL1}.
\]

Move them to:

```text
legacy_debug.representation
```

Do not put them in the main paper result table.

---

# 28. Phase diagnostics

Do not invent a new "phase score".

For finite-horizon chart phase:

\[
u,v\in\{0,\ldots,K-1\}
\]

and phase distance:

\[
d=u+v.
\]

For each phase collect NVR or violation statistics.

Report:

- phase heatmap;
- Pearson \(r\);
- Spearman \(\rho\).

For the current K=4 B8-style audit:

\[
u,v\in\{0,1,2,3\}.
\]

The scientifically safe statement is:

> invalid recovered variation changes systematically with cumulative chart phase.

Do not claim causal propagation from the origin solely from correlation.

---

# 29. Boundary / partition-origin diagnostics

Do not create custom "boundary scores".

For paired image-level error:

\[
\Delta_i
=
E_i^{\mathrm{straddle}}
-
E_i^{\mathrm{interior}}.
\]

Report:

- mean \(\Delta\);
- median \(\Delta\);
- fraction \(\Delta>0\);
- paired bootstrap 95% confidence interval.

Do the same for aligned versus offset partitions.

This is more defensible than another bespoke scalar metric.

---

# 30. Tests: `tests/test_counting_metrics.py`

Create:

```python
import math

import numpy as np

from hpc.metrics.counting import (
    benchmark_count_summary,
    compute_mae,
    compute_nae,
    compute_rmse,
)


def test_mae_rmse_exact():
    pred = np.asarray(
        [1.0, 4.0, 7.0]
    )
    gt = np.asarray(
        [2.0, 2.0, 8.0]
    )

    # errors = [-1, +2, -1]
    assert math.isclose(
        compute_mae(pred, gt),
        4.0 / 3.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        compute_rmse(pred, gt),
        math.sqrt(2.0),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_nae_excludes_zero_gt():
    pred = np.asarray(
        [100.0, 8.0, 12.0]
    )
    gt = np.asarray(
        [0.0, 10.0, 10.0]
    )

    # zero-GT first sample is excluded
    # remaining relative errors: .2, .2
    assert math.isclose(
        compute_nae(pred, gt),
        0.2,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_nae_all_zero_returns_nan():
    pred = np.asarray(
        [0.0, 1.0]
    )
    gt = np.asarray(
        [0.0, 0.0]
    )

    assert math.isnan(
        compute_nae(pred, gt)
    )


def test_public_summary_has_only_canonical_keys():
    out = benchmark_count_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, 3.0]),
    )

    assert set(out.keys()) == {
        "mae",
        "rmse",
        "nae",
        "num_images",
    }
```

---

# 31. Tests: `tests/test_micf_validity_metrics.py`

```python
import math

import torch

from hpc.metrics.micf_validity import (
    aggregate_validity,
    recovered_measure_validity,
)


def test_valid_measure_has_zero_invalidity():
    y = torch.tensor(
        [[[[0.0, 1.0],
           [2.0, 3.0]]]],
        dtype=torch.float32,
    )

    row = recovered_measure_validity(
        y
    )

    assert row.violating_cells == 0
    assert row.vr_tau == 0.0
    assert row.negative_variation == 0.0
    assert row.nvr == 0.0


def test_nvr_is_negative_over_positive_variation():
    y = torch.tensor(
        [[[[2.0, -1.0],
           [0.0, 3.0]]]],
        dtype=torch.float32,
    )

    row = recovered_measure_validity(
        y
    )

    # P = 2 + 3 = 5
    # Q = 1
    assert math.isclose(
        row.positive_variation,
        5.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        row.negative_variation,
        1.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        row.nvr,
        1.0 / 5.0,
        abs_tol=1e-12,
    )


def test_vr_uses_tau():
    y = torch.tensor(
        [[[[0.0, -1e-7],
           [-2e-6, 1.0]]]],
        dtype=torch.float64,
    )

    row = recovered_measure_validity(
        y,
        tau=1e-6,
    )

    assert row.violating_cells == 1
    assert math.isclose(
        row.vr_tau,
        0.25,
        abs_tol=1e-12,
    )


def test_micro_aggregation():
    a = recovered_measure_validity(
        torch.tensor(
            [[[[1.0, -1.0]]]]
        )
    )

    b = recovered_measure_validity(
        torch.tensor(
            [[[[3.0, 0.0]]]]
        )
    )

    out = aggregate_validity(
        [a, b]
    )

    # pooled P = 4
    # pooled Q = 1
    assert math.isclose(
        out["nvr_micro"],
        0.25,
        abs_tol=1e-12,
    )

    # one violating cell out of four
    assert math.isclose(
        out["vr_tau_micro"],
        0.25,
        abs_tol=1e-12,
    )
```

---

# 32. Tests: `tests/test_decomposition_metrics.py`

```python
import math

import numpy as np

from hpc.metrics.decomposition import (
    direct_tiled_discrepancy,
)


def test_zero_discrepancy():
    d = np.asarray(
        [10.0, 20.0]
    )

    t = np.asarray(
        [10.0, 20.0]
    )

    g = np.asarray(
        [10.0, 20.0]
    )

    out = direct_tiled_discrepancy(
        d,
        t,
        g,
    )

    assert (
        out[
            "mean_abs_prediction_discrepancy"
        ]
        == 0.0
    )

    assert (
        out[
            "mean_normalized_prediction_discrepancy"
        ]
        == 0.0
    )


def test_normalized_discrepancy():
    d = np.asarray(
        [12.0, 20.0]
    )

    t = np.asarray(
        [10.0, 25.0]
    )

    g = np.asarray(
        [10.0, 20.0]
    )

    out = direct_tiled_discrepancy(
        d,
        t,
        g,
    )

    # absolute discrepancies = [2,5]
    assert math.isclose(
        out[
            "mean_abs_prediction_discrepancy"
        ],
        3.5,
        abs_tol=1e-12,
    )

    # normalized = [.2,.25]
    assert math.isclose(
        out[
            "mean_normalized_prediction_discrepancy"
        ],
        0.225,
        abs_tol=1e-12,
    )
```

---

# 33. Tests: `tests/test_game_metrics.py`

```python
import math

import numpy as np
import torch

from hpc.metrics.game import (
    game_errors_one_image,
    region_count_from_cell_measure,
)


def test_full_image_region_preserves_all_edge_mass():
    # image is not divisible by stride
    # H=W=10, stride=8 => output grid 2x2
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    total = region_count_from_cell_measure(
        y,
        x0=0.0,
        y0=0.0,
        x1=10.0,
        y1=10.0,
        image_h=10,
        image_w=10,
        stride=8,
    )

    assert math.isclose(
        total,
        10.0,
        abs_tol=1e-12,
    )


def test_game0_equals_absolute_count_error():
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    # 8 GT points anywhere inside image
    points = np.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [9.0, 9.0],
        ],
        dtype=np.float64,
    )

    out = game_errors_one_image(
        y,
        points,
        image_h=10,
        image_w=10,
        stride=8,
        levels=(0,),
    )

    # predicted total = 10
    # GT total = 8
    assert math.isclose(
        out[0],
        2.0,
        abs_tol=1e-12,
    )


def test_game_mass_partition_conservation():
    y = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    # Check predicted region counts across 2x2 GAME
    total = 0.0

    for r in range(2):
        for c in range(2):
            total += region_count_from_cell_measure(
                y,
                x0=5.0 * c,
                y0=5.0 * r,
                x1=5.0 * (c + 1),
                y1=5.0 * (r + 1),
                image_h=10,
                image_w=10,
                stride=8,
            )

    assert math.isclose(
        total,
        10.0,
        abs_tol=1e-10,
    )
```

---

# 34. Cumulative-transform test

Add or preserve a test proving exact inversion.

```python
import torch

from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
)


def test_cumulative_mobius_inversion():
    y = torch.tensor(
        [[[[0.0, 2.0, 1.0],
           [3.0, 0.0, 4.0],
           [1.0, 2.0, 0.0]]]],
        dtype=torch.float64,
    )

    c = cell_counts_to_cumulative_field(
        y
    )

    y_recovered = discrete_mixed_difference(
        c
    )

    assert torch.allclose(
        y,
        y_recovered,
        atol=1e-12,
        rtol=0.0,
    )


def test_count_conservation():
    y = torch.rand(
        1,
        1,
        8,
        9,
        dtype=torch.float64,
    )

    c = cell_counts_to_cumulative_field(
        y
    )

    assert torch.allclose(
        c[..., -1, -1],
        y.sum(
            dim=(-1, -2)
        ),
        atol=1e-12,
        rtol=0.0,
    )
```

---

# 35. Agent migration checklist

The coding agent must perform these exact changes.

## P0 — benchmark correctness

- [ ] Add `benchmark_count_summary`.
- [ ] Fix NAE zero-GT handling.
- [ ] Ensure MAE/RMSE use one prediction per full test image.
- [ ] Main evaluator output uses full-image direct prediction.
- [ ] Do not expose crop MAE as benchmark MAE.
- [ ] Rename rooted squared error to RMSE in our outputs.

## P1 — cumulative validity correctness

- [ ] Add `micf_validity.py`.
- [ ] Replace old NMR denominator `sum(abs(Y))`.
- [ ] Implement:
  \[
  \mathrm{NVR}=Q/P.
  \]
- [ ] Implement `VR_tau`.
- [ ] Canonical paper values are micro VR and micro NVR.
- [ ] Keep macro variants only as supplementary/debugging.

## P2 — decomposition analysis

- [ ] Add paired direct/tiled discrepancy.
- [ ] Do not use aggregate `Direct MAE - Tiled MAE` as the discrepancy metric.

## P3 — GAME correctness

- [ ] Implement actual-support normalization for clipped edge cells.
- [ ] Assert GAME(0) = MAE.
- [ ] Keep pixel-space GAME only.
- [ ] Remove stride-space GAME from paper-facing output.

## P4 — output cleanup

- [ ] Nested JSON namespaces.
- [ ] No PMAE aliases in canonical output.
- [ ] No SRE in canonical output.
- [ ] No C-NMAE/Y-NL1 in canonical output.
- [ ] Conservation is sanity, not performance.

---

# 36. Required command sequence

From repository root:

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m pytest `
  tests/test_counting_metrics.py `
  tests/test_micf_validity_metrics.py `
  tests/test_decomposition_metrics.py `
  tests/test_game_metrics.py `
  -q
```

Then run any existing MICF tests:

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m pytest tests -q
```

Then evaluate B8:

```powershell
.venv\Scripts\python tools/eval_micf_comprehensive.py `
  --checkpoint runs/pilot_micf/b8/best.pt `
  --config configs/pilot_micf/b8.yaml `
  --device cuda `
  --halo 64 `
  --controlled-halo 0
```

Then evaluate PS-FH:

```powershell
.venv\Scripts\python tools/eval_micf_comprehensive.py `
  --checkpoint runs/pilot_micf/psfh_b8_k4/best.pt `
  --config configs/pilot_micf/psfh_b8_k4.yaml `
  --device cuda `
  --halo 64 `
  --controlled-halo 0
```

Use the actual checkpoint/config paths present in the run directory. Do not guess missing paths.

---

# 37. Acceptance gates

The implementation is accepted only if every gate passes.

## Gate A — benchmark formulas

Known-array unit tests must reproduce exact MAE/RMSE/NAE.

---

## Gate B — zero-count NAE

Zero-GT samples must be excluded from NAE.

No `1e12`-scale NAE values from `gt=0` are allowed.

---

## Gate C — NVR definition

For:

\[
Y=[2,-1,0,3]
\]

the evaluator must return:

\[
P=5,\qquad Q=1,\qquad NVR=0.2.
\]

If it returns:

\[
1/6
\]

then the implementation is still using the old \(Q/\sum|Y|\) formula and is wrong.

---

## Gate D — GAME edge conservation

For non-stride-divisible image dimensions:

\[
\sum_r \hat N_r^{GAME(L)}
=
\sum_{ij}\hat Y_{ij}.
\]

---

## Gate E — GAME(0)

Across the dataset:

\[
|\mathrm{GAME}(0)-\mathrm{MAE}|
\]

must be within numerical tolerance.

---

## Gate F — cumulative conservation

For every cumulative prediction:

\[
\hat C[-1,-1]
\approx
\sum\Delta_{xy}\hat C.
\]

Failure is an implementation error.

---

## Gate G — output namespace

`summary.json` must visually separate:

```text
benchmark
method_validity
decomposition
spatial_optional
sanity
protocol
```

A reader must not be able to mistake NVR or a tile diagnostic for a universal crowd-counting benchmark.

---

# 38. Paper-facing result tables after migration

## 38.1 Main SOTA / lightweight table

Use:

| Method | Params | FLOPs | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|
| Baseline | ... | ... | ... | ... |
| PS-FH-CMICF | ... | ... | ... | ... |

Optional NAE column for datasets where it is important.

---

## 38.2 Method-mechanism table

Use:

| Model | MAE ↓ | RMSE ↓ | VR\(_{10^{-6}}\) ↓ | NVR ↓ | Direct-Tiled Norm. Discrepancy ↓ |
|---|---:|---:|---:|---:|---:|
| B8 FH-CMICF | ... | ... | ... | ... | ... |
| PS-FH-CMICF | ... | ... | ... | ... | ... |

Do not compare the NVR column with unrelated SOTA papers as if it were a standard metric.

---

# 39. Important historical-result warning

Existing reported B8/B9 quantities such as:

- B8 Direct NMR ≈ 1.47%;
- B9 Direct NMR ≈ 12.49%;

were produced under the old evaluator's negative-mass-ratio definition.

After changing:

\[
\frac{Q}{P+Q}
\quad\rightarrow\quad
\frac{Q}{P},
\]

the numerical values will change.

Therefore:

\[
\boxed{
\text{Do not mix old NMR numbers with new NVR numbers.}
}
\]

Re-evaluate B5b/B8/B9/PS-FH using the canonical evaluator before constructing the final paper table.

Similarly, old macro VR and new micro VR must not be mixed without labels.

---

# 40. Efficiency metrics

Efficiency is a separate evaluation axis.

Report:

- parameters;
- FLOPs/MACs at an explicitly stated input resolution;
- latency on explicitly stated hardware;
- batch size;
- precision mode.

For comparison with recent lightweight work, a fixed HD resolution such as:

\[
1920\times1080
\]

is useful if the profiling tool actually executes the complete model at that resolution.

Do not derive paper FLOPs from a profiler that silently ignores custom cumulative/integral operations.

The profiler must either:

1. count those operations correctly; or
2. state exactly which operations are omitted.

This is outside the benchmark counting metric implementation and must remain a separate profiler module.

---

# 41. References underlying the metric policy

Use these as methodological references in the paper/repository comments where appropriate.

1. **Zhang et al., CVPR 2016, MCNN**  
   Standard ShanghaiTech crowd-counting evaluation with MAE and rooted squared error historically called MSE.

2. **Wang et al., TPAMI, NWPU-Crowd**  
   Official crowd-counting benchmark uses MAE, rooted MSE/RMSE, and NAE; zero-count images are excluded from NAE.

3. **Tian et al., TIP 2020, PaDNet**  
   Defines PMAE/PRMSE under its own patch protocol; therefore our arbitrary fixed tile diagnostics must not silently reuse those names.

4. **GAME counting metric literature**  
   Spatially partitions the image and satisfies GAME(0)=MAE.

5. **Bivariate cumulative/distribution theory**  
   Valid cumulative surfaces require non-negative rectangle increments / 2-increasingness.

6. **Jordan decomposition of signed measures**  
   A signed measure decomposes into positive and negative variations, motivating explicit positive and negative variation masses for NVR.

---

# 42. Final design rule for agents

The evaluator should answer three different scientific questions, and its code/output structure must preserve the distinction.

### Question 1 — Does the model count accurately relative to other papers?

Use:

\[
\boxed{\mathrm{MAE},\mathrm{RMSE},\mathrm{NAE}}
\]

on whole test images.

### Question 2 — Does a cumulative prediction represent a geometrically valid non-negative counting measure?

Use:

\[
\boxed{\mathrm{VR}_{\tau},\mathrm{NVR}}.
\]

### Question 3 — Is the prediction stable when the same image is evaluated by direct versus tiled decomposition?

Use paired:

\[
\boxed{
|\hat N^D-\hat N^T|
}
\]

and normalized paired discrepancy.

Do not create one giant metric table that mixes these three questions.

---

# 43. Definition of done

The agent is done only when:

1. all new tests pass;
2. existing tests still pass;
3. B8 can be re-evaluated;
4. PS-FH can be re-evaluated;
5. `summary.json` uses the new schema;
6. NAE excludes zero-GT images;
7. NVR uses negative/positive variation;
8. GAME(0)=MAE;
9. whole-image Direct MAE/RMSE are visually obvious as the main benchmark values;
10. no paper-facing output calls our internal tile metric PMAE/PRMSE;
11. old NMR results are explicitly marked legacy and are not mixed with new NVR;
12. no change is made to unrelated train/val/test protocol as part of this evaluator patch.

---

## Short instruction to the coding agent

Implement this specification exactly. Reuse existing model inference and discrete mixed-difference code. Do not redesign the model, alter the dataset split, or change training behavior. The task is to make evaluation definitions, aggregation, naming, tests, and output scientifically correct and clearly separable into benchmark versus method-specific diagnostics.
