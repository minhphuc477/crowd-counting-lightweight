# MICF Comprehensive Checkpoint Evaluation
## Window / Tiled / Direct + Count, Locality, Relative-Error, Tail, and MICF-Validity Metrics

**Repository:** `minhphuc477/crowd-counting-lightweight`  
**Branch:** `MICF`  
**Purpose:** Evaluate an already-trained B5b/B8/MICF checkpoint **without retraining**.

---

# 0. What this evaluator reports

The previous version emphasized only:

\[
\text{Window-MAE},\qquad
\text{Full-Tiled MAE},\qquad
\text{Full-Direct MAE}.
\]

That is not enough. The updated evaluator reports five metric families.

| Family | Metrics |
|---|---|
| **Standard image-level counting** | MAE, RMSE (often called “MSE” in crowd-counting papers), NAE, SRE |
| **Local / patch counting** | Window-PMAE, Window-PRMSE, non-zero-window NAE, empty-window error, non-empty-window error, GAME(0–3) |
| **Error distribution / calibration** | signed bias, median AE, P90 AE, P95 AE, max AE |
| **MICF measure validity** | violation rate, violation magnitude, negative-mass ratio, negative-mass total, count↔mass conservation error |
| **Representation diagnostics** | normalized cumulative-field MAE, normalized local-measure L1 error, cancellation ratio, Direct−Tiled gaps |

This is intentionally broader than the final paper headline table.  
The evaluator should **compute and save everything**, while the paper can later select the relevant metrics.

---

# 1. Standard image-level metrics

For \(M\) test images, predicted count \(\hat N_i\) and GT count \(N_i\):

## 1.1 MAE

\[
\boxed{
MAE=
\frac1M\sum_{i=1}^{M}
|\hat N_i-N_i|
}
\]

Measures average count accuracy.

---

## 1.2 RMSE

\[
\boxed{
RMSE=
\sqrt{
\frac1M
\sum_{i=1}^{M}
(\hat N_i-N_i)^2
}
}
\]

Many crowd-counting papers label this column **MSE**, although the reported formula contains the square root.  
The code uses the mathematically correct name `rmse`.

RMSE exposes catastrophic failures more strongly than MAE.

---

## 1.3 NAE — Normalized Absolute Error

\[
\boxed{
NAE=
\frac1M
\sum_{i=1}^{M}
\frac{|\hat N_i-N_i|}{N_i}
}
\]

This measures error relative to scene crowd size.

Implementation uses:

\[
\max(N_i,\epsilon)
\]

for numerical safety.

For ShanghaiTech full images, counts are positive, so this does not normally alter the metric.

---

## 1.4 SRE — Squared Relative Error

Following crowd-counting evaluation literature:

\[
\boxed{
SRE=
\sqrt{
\frac1M
\sum_{i=1}^{M}
\frac{(\hat N_i-N_i)^2}{N_i}
}
}
\]

Again the denominator is protected by \(\epsilon\).

SRE penalizes large errors while accounting for the GT scale.

---

# 2. Three inference regimes

The same checkpoint is evaluated in three regimes.

## 2.1 Window regime

Every test image is covered by non-overlapping \(256\times256\) core windows.

For image \(i\), window \(j\):

\[
e_{ij}
=
\hat N_{ij}-N_{ij}.
\]

Primary local metric:

\[
\boxed{
PMAE
=
\frac{
\sum_i\sum_j|e_{ij}|
}{
\sum_iK_i
}
}
\]

This evaluator names the same quantity:

```text
window_mae
window_pmae
```

Likewise:

\[
\boxed{
PRMSE
=
\sqrt{
\frac{
\sum_i\sum_j e_{ij}^2
}{
\sum_iK_i
}
}
}
\]

saved as:

```text
window_rmse
window_prmse
```

PMAE/PRMSE correspond to the patch-level evaluation idea used in PaDNet.

---

## 2.2 Full-Tiled regime

The image is divided into local cores, each predicted with halo context.

For MICF:

\[
\hat C^{tile}
\rightarrow
\hat Y^{tile}=\Delta_{xy}\hat C^{tile},
\]

crop the valid core measure, stitch all cores:

\[
\hat Y^G
=
\operatorname{Stitch}
(\hat Y^{core}_1,\ldots,\hat Y^{core}_{K_i}),
\]

then:

\[
\hat C^G=P\hat Y^G.
\]

Whole-image tiled count:

\[
\boxed{
\hat N_i^{tile}
=
\hat C^G[-1,-1]
}
\]

and all image-level metrics are computed from \(\hat N_i^{tile}\).

---

## 2.3 Full-Direct regime

One forward pass:

\[
I_i
\rightarrow
\hat C_i^D
\]

with:

\[
\boxed{
\hat N_i^D=\hat C_i^D[-1,-1].
}
\]

Again compute MAE, RMSE, NAE, SRE, bias and tail statistics.

---

# 3. Window metrics beyond MAE/RMSE

## 3.1 Non-zero-window NAE

Window GT can be zero, so ordinary NAE is undefined for empty windows.

Therefore:

\[
\boxed{
NAE_{window,+}
=
\frac1{|\mathcal W_+|}
\sum_{(i,j)\in\mathcal W_+}
\frac{|e_{ij}|}{N_{ij}}
}
\]

where:

\[
\mathcal W_+
=
\{(i,j):N_{ij}>0\}.
\]

Saved as:

```text
window_nae_nonzero
```

---

## 3.2 Empty-window error

For:

\[
N_{ij}=0,
\]

report:

\[
\boxed{
E_{empty}
=
\operatorname{mean}_{N_{ij}=0}
|\hat N_{ij}|
}
\]

Saved as:

```text
empty_window_mae
empty_window_mean_prediction
empty_window_fraction
```

This directly exposes false crowd mass in background regions.

---

## 3.3 Non-empty-window MAE

\[
\boxed{
MAE_{nonempty}
=
\operatorname{mean}_{N_{ij}>0}
|\hat N_{ij}-N_{ij}|
}
\]

Saved as:

```text
nonempty_window_mae
```

This separates foreground counting from background hallucination.

---

# 4. GAME — Grid Average Mean Absolute Error

Global MAE can hide spatially incorrect mass.

At GAME level \(L\), partition the output count grid into:

\[
2^L\times2^L=4^L
\]

non-overlapping regions.

For image \(i\):

\[
GAME_i(L)
=
\sum_{r=1}^{4^L}
|\hat N_{ir}-N_{ir}|.
\]

Dataset-level:

\[
\boxed{
GAME(L)
=
\frac1M
\sum_{i=1}^{M}
GAME_i(L)
}
\]

The evaluator computes:

```text
GAME(0)
GAME(1)
GAME(2)
GAME(3)
```

for both:

```text
direct
tiled
```

Important:

- `GAME(0)` is the whole-image spatial-measure error using **in-bounds GT**.
- Larger \(L\) increasingly penalizes spatial misallocation.
- MICF is stride-16 in the current pilot, so GAME is evaluated on the reconstructed stride-level count measure \(Y\). It is a valid MICF spatial diagnostic, but should not be compared numerically to a full-resolution density-map GAME result without noting the grid resolution.

---

# 5. Signed bias and tail statistics

Mean error:

\[
\boxed{
Bias
=
\frac1M
\sum_i(\hat N_i-N_i)
}
\]

Interpretation:

- \(Bias>0\): systematic over-counting.
- \(Bias<0\): systematic under-counting.

The evaluator also reports absolute-error distribution:

\[
MedianAE,
\quad
P90AE,
\quad
P95AE,
\quad
MaxAE.
\]

These are not primary benchmark metrics, but they reveal whether a similar MAE is caused by:

- many moderate errors, or
- a few catastrophic failures.

They are computed for:

```text
Window
Full-Tiled
Full-Direct
```

---

# 6. Local error cancellation

For image \(i\):

\[
E_i^{local}
=
\sum_j|e_{ij}|
\]

while the net local-window count error is:

\[
E_i^{net}
=
\left|\sum_je_{ij}\right|.
\]

Triangle inequality:

\[
\boxed{
E_i^{net}
\le E_i^{local}.
}
\]

Define:

\[
\boxed{
R_i^{cancel}
=
1-
\frac{
|\sum_je_{ij}|
}{
\sum_j|e_{ij}|+\epsilon
}
}
\]

Interpretation:

- \(0\): local errors mostly have the same sign; little cancellation.
- close to \(1\): over/under-counting cancels strongly at image level.

Saved as:

```text
mean_cancellation_ratio
median_cancellation_ratio
p90_cancellation_ratio
```

This is a diagnostic, not a standard crowd-counting benchmark metric.

---

# 7. MICF measure-validity metrics

MICF reconstructs a local measure:

\[
\hat Y=\Delta_{xy}\hat C.
\]

A valid count measure requires:

\[
\hat Y_{uv}\ge0.
\]

The evaluator reports the following separately for direct and tiled outputs.

## 7.1 Violation rate

\[
\boxed{
VR
=
\frac{
\#\{(u,v):\hat Y_{uv}<0\}
}{
H_oW_o
}
}
\]

Saved as:

```text
direct_violation_rate
tiled_violation_rate
```

---

## 7.2 Violation magnitude

\[
\boxed{
VM
=
\frac1{H_oW_o}
\sum_{uv}
[-\hat Y_{uv}]_+
}
\]

Saved as:

```text
direct_violation_magnitude
tiled_violation_magnitude
```

---

## 7.3 Negative-mass ratio

\[
\boxed{
NMR
=
\frac{
\sum_{uv}[-\hat Y_{uv}]_+
}{
\sum_{uv}|\hat Y_{uv}|+\epsilon
}
}
\]

Saved as:

```text
direct_negative_mass_ratio
tiled_negative_mass_ratio
```

---

## 7.4 Total negative mass

\[
\boxed{
M_-=
\sum_{uv}[-\hat Y_{uv}]_+
}
\]

Saved as:

```text
direct_negative_mass_total
tiled_negative_mass_total
```

Unlike violation magnitude, this reveals the absolute amount of invalid negative count mass.

---

# 8. Count ↔ measure conservation

For a cumulative field:

\[
N_C=\hat C[-1,-1].
\]

Mixed difference should satisfy:

\[
N_Y=\sum_{uv}\Delta_{xy}\hat C_{uv}.
\]

Therefore:

\[
\boxed{
E_{cons}
=
|N_C-N_Y|.
}
\]

Saved as:

```text
direct_conservation_error
tiled_conservation_error
```

This should be near floating-point zero.

A material non-zero value indicates an implementation/composition/cropping error.

This is an **integrity check**, not a model-quality metric.

---

# 9. MICF representation diagnostics

The whole-image count can be correct while the cumulative field is wrong internally.

Therefore two representation-specific diagnostics are added.

## 9.1 Normalized cumulative-field MAE

GT cumulative field:

\[
C=P Y.
\]

Define:

\[
\boxed{
C\text{-NMAE}
=
\frac1{H_oW_o}
\sum_{uv}
\left|
\frac{
\hat C_{uv}-C_{uv}
}{
\max(N,1)
}
\right|
}
\]

Saved as:

```text
direct_cumulative_field_nmae
tiled_cumulative_field_nmae
```

This uses one image-level normalization scalar, consistent with the B5b training-loss philosophy.

---

## 9.2 Normalized local-measure L1

\[
\boxed{
Y\text{-NL1}
=
\frac{
\sum_{uv}
|\hat Y_{uv}-Y_{uv}|
}{
\max(N,1)
}
}
\]

Saved as:

```text
direct_measure_nl1
tiled_measure_nl1
```

Interpretation:

- low count MAE + high `Y-NL1` means total count may be correct through spatial error cancellation;
- low `Y-NL1` means the recovered count measure itself is spatially closer to GT.

---

# 10. Direct-vs-Tiled gaps

Report:

\[
\boxed{
G_{MAE}
=
MAE_{direct}-MAE_{tile}
}
\]

\[
\boxed{
G_{RMSE}
=
RMSE_{direct}-RMSE_{tile}
}
\]

\[
\boxed{
G_{NAE}
=
NAE_{direct}-NAE_{tile}
}
\]

and:

\[
\boxed{
G_{SRE}
=
SRE_{direct}-SRE_{tile}.
}
\]

Large positive gaps with good local metrics support a direct/global extrapolation bottleneck.

---

# 11. Metrics deliberately NOT added

## 11.1 PSNR / SSIM

These are useful when evaluating a predicted **density image** against a GT density image.

MICF predicts a cumulative count field and its mixed difference is an exact count measure, not a Gaussian density image.

Therefore PSNR/SSIM would mix representation semantics and would not be directly comparable to density-map papers.

Do not add them to this evaluator unless the project later defines a principled common raster target.

---

## 11.2 Localization Precision / Recall / F1

These require predicted person locations and a matching protocol.

The present evaluator is count/measure evaluation only.

Do not fabricate localization metrics from MICF unless a localization decoder/evaluator is explicitly being tested.

---

# 12. Literature status of the metrics

The following are established crowd-counting evaluation ideas and must **not** be claimed as novel:

- MAE / RMSE;
- NAE;
- SRE;
- GAME;
- PMAE / PRMSE.

The MICF-specific use of:

- direct-vs-tiled gap;
- cumulative-field normalized error;
- mixed-difference validity;
- cancellation ratio;

should be framed as **diagnostic analysis of the representation**, not as a claim that the underlying mathematical quantities themselves are new.

---

# 13. Full evaluator code

Create:

```text
tools/eval_micf_comprehensive.py
```

```python
"""Comprehensive MICF checkpoint evaluator.

Evaluates one already-trained checkpoint without updating weights.

Metric families
---------------
1. Standard image-level:
   MAE, RMSE, NAE, SRE, signed bias, Median/P90/P95/Max AE.

2. Local/window:
   PMAE/PRMSE, non-zero-window NAE, empty-window error,
   non-empty-window MAE, local cancellation.

3. Spatial:
   GAME(0..3) on the reconstructed stride-level count measure.

4. MICF validity:
   violation rate, violation magnitude, negative-mass ratio,
   total negative mass, count<->measure conservation error.

5. MICF representation:
   normalized cumulative-field MAE and normalized local-measure L1.

6. Three inference regimes:
   Window, Full-Tiled, Full-Direct.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.sha import ShanghaiTechDataset
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
    points_to_count_map,
)
from hpc.models.micf_lite import (
    MICFLite,
    compose_tiled_cumulative_field,
)


# ---------------------------------------------------------------------
# CLI / loading
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive MICF checkpoint evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML fallback. Normally the checkpoint already "
            "contains its config."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_data",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--halo",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--direct-pad-multiple",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--game-levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--verify-repo-tiled",
        action="store_true",
    )
    parser.add_argument(
        "--strict-tiled-tolerance",
        type=float,
        default=1e-3,
    )
    return parser.parse_args()


def safe_torch_load(
    path: str,
    map_location: str | torch.device = "cpu",
) -> dict:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_config(
    checkpoint: dict,
    config_path: str | None,
) -> dict:
    cfg = checkpoint.get("config")
    if cfg is not None:
        return cfg

    if config_path is None:
        raise ValueError(
            "Checkpoint has no stored config. "
            "Provide --config."
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def build_model_from_config(
    cfg: dict,
    state_dict: dict,
    device: torch.device,
) -> MICFLite:
    m_cfg = cfg.get("model", {})

    model = MICFLite(
        backbone_name=m_cfg.get(
            "backbone",
            "mobilenetv4_conv_small_050.e3000_r224_in1k",
        ),
        pretrained=False,
        neck_width=int(
            m_cfg.get("neck_width", 32)
        ),
        context_dilations=tuple(
            m_cfg.get(
                "context_dilations",
                [1, 2, 3],
            )
        ),
        use_integral_context=bool(
            m_cfg.get(
                "use_integral_context",
                False,
            )
        ),
        context_type=str(
            m_cfg.get(
                "context_type",
                "directional",
            )
        ),
        head_type=m_cfg.get(
            "head_type",
            "cumulative",
        ),
        output_stride=int(
            m_cfg.get("output_stride", 16)
        ),
        eps_d=float(
            m_cfg.get("eps_d", 1e-8)
        ),
        extent_aware=bool(
            m_cfg.get(
                "extent_aware",
                False,
            )
        ),
        finite_horizon=m_cfg.get(
            "finite_horizon",
            None,
        ),
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    return model.to(device).eval()


def build_dataset(
    cfg: dict,
    root_override: str | None,
    part_override: str | None,
    split: str,
) -> ShanghaiTechDataset:
    ds_cfg = cfg.get("dataset", {})

    root = (
        root_override
        if root_override is not None
        else ds_cfg.get(
            "root",
            "./data/ShanghaiTech",
        )
    )
    part = (
        part_override
        if part_override is not None
        else ds_cfg.get(
            "part",
            "part_A",
        )
    )

    return ShanghaiTechDataset(
        root=root,
        part=part,
        split=split,
        crop_size=int(
            ds_cfg.get("crop_size", 256)
        ),
        is_train=False,
        image_mean=ds_cfg.get(
            "image_mean",
            [0.5, 0.5, 0.5],
        ),
        image_std=ds_cfg.get(
            "image_std",
            [0.5, 0.5, 0.5],
        ),
        coordinate_base=int(
            ds_cfg.get(
                "coordinate_base",
                0,
            )
        ),
        annotation_bounds_policy="allow",
    )


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def as_numpy_points(
    points: Any,
) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        arr = (
            points.detach()
            .cpu()
            .numpy()
        )
    else:
        arr = np.asarray(points)

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    if arr.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return arr.reshape(-1, 2)


def in_bounds_points(
    points: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if len(points) == 0:
        return points

    mask = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    return points[mask]


def count_points_in_window(
    points: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> int:
    if len(points) == 0:
        return 0

    mask = (
        (points[:, 0] >= float(x0))
        & (points[:, 0] < float(x1))
        & (points[:, 1] >= float(y0))
        & (points[:, 1] < float(y1))
    )
    return int(mask.sum())


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    keys: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(
    values: Iterable[float],
) -> float:
    arr = np.asarray(
        list(values),
        dtype=np.float64,
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_percentile(
    values: Iterable[float],
    q: float,
) -> float:
    arr = np.asarray(
        list(values),
        dtype=np.float64,
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


# ---------------------------------------------------------------------
# Standard count metrics
# ---------------------------------------------------------------------

def count_metric_summary(
    predictions: Iterable[float],
    targets: Iterable[float],
    eps: float = 1e-12,
) -> dict[str, float]:
    pred = np.asarray(
        list(predictions),
        dtype=np.float64,
    )
    gt = np.asarray(
        list(targets),
        dtype=np.float64,
    )

    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: "
            f"{pred.shape} vs {gt.shape}"
        )

    if pred.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "nae": float("nan"),
            "sre": float("nan"),
            "signed_bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "p95_ae": float("nan"),
            "max_ae": float("nan"),
        }

    err = pred - gt
    ae = np.abs(err)

    denom = np.maximum(
        gt,
        eps,
    )

    return {
        "mae": float(np.mean(ae)),
        "rmse": float(
            np.sqrt(
                np.mean(err * err)
            )
        ),
        "nae": float(
            np.mean(
                ae / denom
            )
        ),
        "sre": float(
            np.sqrt(
                np.mean(
                    (err * err)
                    / denom
                )
            )
        ),
        "signed_bias": float(
            np.mean(err)
        ),
        "median_ae": float(
            np.median(ae)
        ),
        "p90_ae": float(
            np.percentile(
                ae,
                90,
            )
        ),
        "p95_ae": float(
            np.percentile(
                ae,
                95,
            )
        ),
        "max_ae": float(
            np.max(ae)
        ),
    }


def window_metric_summary(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    if not rows:
        return {}

    pred = np.asarray(
        [
            float(r["pred_count"])
            for r in rows
        ],
        dtype=np.float64,
    )
    gt = np.asarray(
        [
            float(r["gt_count"])
            for r in rows
        ],
        dtype=np.float64,
    )

    stats = count_metric_summary(
        pred,
        gt,
        eps=1.0,
    )

    # Ordinary NAE is not meaningful for GT=0 windows.
    positive = gt > 0
    empty = gt == 0

    if positive.any():
        nae_nonzero = float(
            np.mean(
                np.abs(
                    pred[positive]
                    - gt[positive]
                )
                / gt[positive]
            )
        )
        nonempty_mae = float(
            np.mean(
                np.abs(
                    pred[positive]
                    - gt[positive]
                )
            )
        )
    else:
        nae_nonzero = float("nan")
        nonempty_mae = float("nan")

    if empty.any():
        empty_mae = float(
            np.mean(
                np.abs(
                    pred[empty]
                )
            )
        )
        empty_mean_pred = float(
            np.mean(
                pred[empty]
            )
        )
    else:
        empty_mae = float("nan")
        empty_mean_pred = float("nan")

    stats.update(
        {
            "pmae": stats["mae"],
            "prmse": stats["rmse"],
            "nae_nonzero": nae_nonzero,
            "nonempty_window_mae": (
                nonempty_mae
            ),
            "empty_window_mae": (
                empty_mae
            ),
            "empty_window_mean_prediction": (
                empty_mean_pred
            ),
            "empty_window_fraction": float(
                empty.mean()
            ),
        }
    )

    # Remove the arbitrary eps-protected window NAE/SRE aliases.
    # Keep only the explicitly meaningful non-zero-window NAE.
    stats.pop("nae", None)
    stats.pop("sre", None)

    return stats


# ---------------------------------------------------------------------
# MICF / measure metrics
# ---------------------------------------------------------------------

def measure_validity_metrics(
    y_pred: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    y = y_pred.float()

    negative = (
        -y
    ).clamp_min(0.0)

    neg_total = float(
        negative.sum().item()
    )
    abs_total = float(
        y.abs().sum().item()
    )

    return {
        "violation_rate": float(
            (y < 0).float()
            .mean()
            .item()
        ),
        "violation_magnitude": float(
            negative.mean().item()
        ),
        "negative_mass_ratio": float(
            neg_total
            / (abs_total + eps)
        ),
        "negative_mass_total": (
            neg_total
        ),
    }


def representation_metrics(
    pred_c: torch.Tensor,
    pred_y: torch.Tensor,
    gt_c: torch.Tensor,
    gt_y: torch.Tensor,
    gt_count: float,
) -> dict[str, float]:
    scale = max(
        float(gt_count),
        1.0,
    )

    c_nmae = float(
        (
            (
                pred_c.float()
                - gt_c.float()
            ).abs()
            / scale
        )
        .mean()
        .item()
    )

    y_nl1 = float(
        (
            pred_y.float()
            - gt_y.float()
        )
        .abs()
        .sum()
        .item()
        / scale
    )

    pred_count_from_c = float(
        pred_c[..., -1, -1]
        .reshape(-1)[0]
        .item()
    )
    pred_count_from_y = float(
        pred_y.sum().item()
    )

    conservation_error = abs(
        pred_count_from_c
        - pred_count_from_y
    )

    return {
        "cumulative_field_nmae": (
            c_nmae
        ),
        "measure_nl1": y_nl1,
        "conservation_error": (
            conservation_error
        ),
    }


# ---------------------------------------------------------------------
# GAME
# ---------------------------------------------------------------------

def game_errors(
    pred_y: torch.Tensor,
    gt_y: torch.Tensor,
    levels: Iterable[int],
) -> dict[int, float]:
    """GAME on the model's stride-level count measure.

    Both tensors must be [1,1,H,W] and spatially aligned.
    """
    if pred_y.shape != gt_y.shape:
        raise ValueError(
            f"GAME shape mismatch: "
            f"{tuple(pred_y.shape)} "
            f"vs {tuple(gt_y.shape)}"
        )

    if pred_y.ndim != 4:
        raise ValueError(
            "GAME expects [1,1,H,W]."
        )

    _, _, H, W = pred_y.shape

    results: dict[int, float] = {}

    for level in levels:
        level = int(level)
        if level < 0:
            raise ValueError(
                "GAME level must be >= 0"
            )

        parts = 2 ** level

        # Integer boundaries cover every cell exactly once.
        y_edges = np.linspace(
            0,
            H,
            parts + 1,
            dtype=np.int64,
        )
        x_edges = np.linspace(
            0,
            W,
            parts + 1,
            dtype=np.int64,
        )

        total = 0.0

        for r in range(parts):
            y0 = int(y_edges[r])
            y1 = int(y_edges[r + 1])

            for c in range(parts):
                x0 = int(x_edges[c])
                x1 = int(x_edges[c + 1])

                pred_count = float(
                    pred_y[
                        ...,
                        y0:y1,
                        x0:x1,
                    ]
                    .sum()
                    .item()
                )
                gt_count = float(
                    gt_y[
                        ...,
                        y0:y1,
                        x0:x1,
                    ]
                    .sum()
                    .item()
                )

                total += abs(
                    pred_count
                    - gt_count
                )

        results[level] = float(total)

    return results


# ---------------------------------------------------------------------
# Tiled/window inference
# ---------------------------------------------------------------------

def validate_tiling_geometry(
    model: MICFLite,
    tile_size: int,
    halo: int,
) -> None:
    if tile_size <= 0:
        raise ValueError(
            "tile_size must be positive"
        )
    if halo < 0:
        raise ValueError(
            "halo must be >= 0"
        )

    stride = int(model.output_stride)

    required = (
        stride
        if model.finite_horizon is None
        else stride
        * int(model.finite_horizon)
    )

    if tile_size % required != 0:
        raise ValueError(
            f"tile_size={tile_size} "
            f"must be divisible by {required}"
        )

    if halo % required != 0:
        raise ValueError(
            f"halo={halo} "
            f"must be divisible by {required}"
        )


@torch.no_grad()
def predict_windows_and_tiled(
    model: MICFLite,
    image: torch.Tensor,
    points_in_bounds: np.ndarray,
    tile_size: int,
    halo: int,
) -> tuple[
    float,
    list[dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
]:
    """Return tiled count, per-window rows, tiled C, tiled Y.

    Cumulative path mirrors MICFLite.predict_tiled():
        halo forward
        -> mixed difference
        -> core Y
        -> local core C
        -> exact global composition
    """
    if (
        image.ndim != 4
        or image.shape[0] != 1
    ):
        raise ValueError(
            f"Expected [1,3,H,W], "
            f"got {tuple(image.shape)}"
        )

    validate_tiling_geometry(
        model,
        tile_size,
        halo,
    )

    _, _, H, W = image.shape
    stride = int(model.output_stride)
    out_tile = tile_size // stride

    n_h = math.ceil(
        H / tile_size
    )
    n_w = math.ceil(
        W / tile_size
    )

    padded_h = n_h * tile_size
    padded_w = n_w * tile_size

    x_pad = F.pad(
        image,
        (
            0,
            padded_w - W,
            0,
            padded_h - H,
        ),
        mode="constant",
        value=0.0,
    )

    is_cumulative = (
        model.head_type
        in {
            "cumulative",
            "integrated_local",
        }
    )

    c_local: (
        list[list[torch.Tensor | None]]
        | None
    ) = None

    y_global_local = torch.zeros(
        (
            1,
            1,
            n_h * out_tile,
            n_w * out_tile,
        ),
        device=image.device,
        dtype=torch.float32,
    )

    if is_cumulative:
        c_local = [
            [None] * n_w
            for _ in range(n_h)
        ]

    rows: list[dict[str, Any]] = []

    for tile_r in range(n_h):
        for tile_c in range(n_w):
            y0 = tile_r * tile_size
            x0 = tile_c * tile_size

            y1_core = (
                tile_r + 1
            ) * tile_size
            x1_core = (
                tile_c + 1
            ) * tile_size

            valid_y1 = min(
                y1_core,
                H,
            )
            valid_x1 = min(
                x1_core,
                W,
            )

            valid_h = (
                valid_y1 - y0
            )
            valid_w = (
                valid_x1 - x0
            )

            hy0 = max(
                0,
                y0 - halo,
            )
            hx0 = max(
                0,
                x0 - halo,
            )
            hy1 = min(
                padded_h,
                y1_core + halo,
            )
            hx1 = min(
                padded_w,
                x1_core + halo,
            )

            crop = x_pad[
                ...,
                hy0:hy1,
                hx0:hx1,
            ]

            field = model.forward_field(
                crop
            )

            ry0 = (
                y0 - hy0
            ) // stride
            rx0 = (
                x0 - hx0
            ) // stride

            if is_cumulative:
                if halo > 0:
                    y_full = (
                        discrete_mixed_difference(
                            field
                        )
                    )
                    y_core = y_full[
                        ...,
                        ry0:(
                            ry0
                            + out_tile
                        ),
                        rx0:(
                            rx0
                            + out_tile
                        ),
                    ]
                else:
                    c_core_raw = field[
                        ...,
                        :out_tile,
                        :out_tile,
                    ]
                    y_core = (
                        discrete_mixed_difference(
                            c_core_raw
                        )
                    )

                c_tile = torch.cumsum(
                    torch.cumsum(
                        y_core,
                        dim=-2,
                    ),
                    dim=-1,
                )

                assert c_local is not None
                c_local[
                    tile_r
                ][tile_c] = (
                    c_tile.squeeze(0)
                    .squeeze(0)
                )
            else:
                y_core = field[
                    ...,
                    ry0:(
                        ry0
                        + out_tile
                    ),
                    rx0:(
                        rx0
                        + out_tile
                    ),
                ].float()

            oy0 = (
                tile_r
                * out_tile
            )
            ox0 = (
                tile_c
                * out_tile
            )

            y_global_local[
                ...,
                oy0:(
                    oy0 + out_tile
                ),
                ox0:(
                    ox0 + out_tile
                ),
            ] = y_core.float()

            valid_out_h = math.ceil(
                valid_h / stride
            )
            valid_out_w = math.ceil(
                valid_w / stride
            )

            pred_window = float(
                y_core[
                    ...,
                    :valid_out_h,
                    :valid_out_w,
                ]
                .sum()
                .item()
            )

            gt_window = (
                count_points_in_window(
                    points_in_bounds,
                    x0=x0,
                    y0=y0,
                    x1=valid_x1,
                    y1=valid_y1,
                )
            )

            signed = (
                pred_window
                - float(gt_window)
            )

            rows.append(
                {
                    "tile_row": tile_r,
                    "tile_col": tile_c,
                    "x0": x0,
                    "y0": y0,
                    "x1": valid_x1,
                    "y1": valid_y1,
                    "valid_width": valid_w,
                    "valid_height": valid_h,
                    "gt_count": gt_window,
                    "pred_count": pred_window,
                    "signed_error": signed,
                    "abs_error": abs(
                        signed
                    ),
                }
            )

    out_h_full = math.ceil(
        H / stride
    )
    out_w_full = math.ceil(
        W / stride
    )

    if is_cumulative:
        assert c_local is not None

        complete: list[
            list[torch.Tensor]
        ] = []

        for row in c_local:
            if any(
                v is None
                for v in row
            ):
                raise RuntimeError(
                    "Incomplete tile grid."
                )

            complete.append(
                [
                    v
                    for v in row
                    if v is not None
                ]
            )

        c_global_2d = (
            compose_tiled_cumulative_field(
                complete
            )
        )

        c_tiled = c_global_2d[
            :out_h_full,
            :out_w_full,
        ].unsqueeze(0).unsqueeze(0)

        y_tiled = (
            discrete_mixed_difference(
                c_tiled
            )
        )
    else:
        y_tiled = y_global_local[
            ...,
            :out_h_full,
            :out_w_full,
        ]
        c_tiled = (
            cell_counts_to_cumulative_field(
                y_tiled,
                orientation="TL",
            )
        )

    pred_tiled = float(
        c_tiled[
            ...,
            -1,
            -1,
        ]
        .reshape(-1)[0]
        .item()
    )

    sum_window_predictions = float(
        sum(
            float(r["pred_count"])
            for r in rows
        )
    )

    if not math.isclose(
        pred_tiled,
        sum_window_predictions,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        raise RuntimeError(
            "Tiled/window prediction mismatch: "
            f"C_corner={pred_tiled:.6f}, "
            f"sum_windows="
            f"{sum_window_predictions:.6f}"
        )

    return (
        pred_tiled,
        rows,
        c_tiled,
        y_tiled,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
    )

    checkpoint_path = Path(
        args.checkpoint
    )
    checkpoint = safe_torch_load(
        str(checkpoint_path),
        map_location="cpu",
    )

    cfg = load_config(
        checkpoint,
        args.config,
    )

    state_dict = checkpoint.get(
        "state_dict"
    )

    if state_dict is None:
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint has no state_dict."
            )

    model = build_model_from_config(
        cfg,
        state_dict,
        device,
    )

    dataset = build_dataset(
        cfg,
        args.dataset_root,
        args.part,
        args.split,
    )

    if args.output_dir is None:
        output_dir = (
            checkpoint_path.parent
            / "eval_comprehensive"
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_eval = len(dataset)
    if args.max_samples is not None:
        n_eval = min(
            n_eval,
            int(args.max_samples),
        )

    direct_predictions: list[float] = []
    tiled_predictions: list[float] = []
    full_gt_counts: list[float] = []

    all_window_rows: list[
        dict[str, Any]
    ] = []
    per_image_rows: list[
        dict[str, Any]
    ] = []

    cancellation_ratios: list[
        float
    ] = []

    direct_game_by_level: dict[
        int,
        list[float],
    ] = {
        int(level): []
        for level in args.game_levels
    }

    tiled_game_by_level: dict[
        int,
        list[float],
    ] = {
        int(level): []
        for level in args.game_levels
    }

    direct_validity_rows: list[
        dict[str, float]
    ] = []
    tiled_validity_rows: list[
        dict[str, float]
    ] = []

    direct_repr_rows: list[
        dict[str, float]
    ] = []
    tiled_repr_rows: list[
        dict[str, float]
    ] = []

    print("=" * 96)
    print(
        "MICF COMPREHENSIVE CHECKPOINT EVALUATION"
    )
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(
        f"Head       : {model.head_type}"
        f" | stride={model.output_stride}"
        f" | FH={model.finite_horizon}"
    )
    print(
        f"Tile       : {args.tile_size}"
        f" | halo={args.halo}"
    )
    print(
        f"GAME       : {args.game_levels}"
    )
    print(
        f"Images     : {n_eval}"
    )
    print("=" * 96)

    for image_index in range(n_eval):
        sample = dataset[image_index]

        image = (
            sample["image"]
            .unsqueeze(0)
            .to(device)
        )

        _, _, H, W = image.shape
        stride = int(
            model.output_stride
        )

        raw_points = as_numpy_points(
            sample["gt_points"]
        )
        points_inside = in_bounds_points(
            raw_points,
            height=H,
            width=W,
        )

        gt_count = float(
            sample["gt_count"].item()
        )
        gt_in_bounds = float(
            len(points_inside)
        )
        gt_out_of_bounds = (
            gt_count
            - gt_in_bounds
        )

        # ---------------------------------------------------------
        # Full-Direct
        # ---------------------------------------------------------
        pred_direct_count, direct_field = (
            model.predict(
                image,
                pad_multiple=(
                    args.direct_pad_multiple
                ),
            )
        )

        pred_direct_count = float(
            torch.as_tensor(
                pred_direct_count
            ).item()
        )

        if (
            model.head_type
            in {
                "cumulative",
                "integrated_local",
            }
        ):
            c_direct = (
                direct_field.float()
            )
            y_direct = (
                discrete_mixed_difference(
                    c_direct
                )
            )
        else:
            y_direct = (
                direct_field.float()
            )
            c_direct = (
                cell_counts_to_cumulative_field(
                    y_direct,
                    orientation="TL",
                )
            )

        # ---------------------------------------------------------
        # Window + Full-Tiled
        # ---------------------------------------------------------
        (
            pred_tiled_count,
            window_rows,
            c_tiled,
            y_tiled,
        ) = predict_windows_and_tiled(
            model=model,
            image=image,
            points_in_bounds=(
                points_inside
            ),
            tile_size=args.tile_size,
            halo=args.halo,
        )

        # ---------------------------------------------------------
        # Shared exact GT count measure on output grid
        # ---------------------------------------------------------
        out_h = c_direct.shape[-2]
        out_w = c_direct.shape[-1]

        if (
            c_tiled.shape[-2:]
            != (out_h, out_w)
        ):
            raise RuntimeError(
                "Direct/tiled output-grid mismatch: "
                f"{c_direct.shape[-2:]} "
                f"vs {c_tiled.shape[-2:]}"
            )

        gt_y_2d = points_to_count_map(
            points_inside,
            out_h=out_h,
            out_w=out_w,
            stride=stride,
            device=device,
            dtype=torch.float32,
        )

        gt_y = (
            gt_y_2d
            .unsqueeze(0)
            .unsqueeze(0)
        )

        gt_c = (
            cell_counts_to_cumulative_field(
                gt_y,
                orientation="TL",
            )
        )

        if not math.isclose(
            float(gt_y.sum().item()),
            gt_in_bounds,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                "GT measure does not conserve "
                "in-bounds count."
            )

        # ---------------------------------------------------------
        # Optional repository tiled equivalence
        # ---------------------------------------------------------
        repo_tiled_count = float("nan")
        repo_tiled_diff = float("nan")

        if args.verify_repo_tiled:
            repo_tiled, _ = (
                model.predict_tiled(
                    image,
                    tile_size=(
                        args.tile_size
                    ),
                    halo=args.halo,
                )
            )

            repo_tiled_count = float(
                torch.as_tensor(
                    repo_tiled
                ).item()
            )
            repo_tiled_diff = abs(
                repo_tiled_count
                - pred_tiled_count
            )

            if (
                repo_tiled_diff
                > args.strict_tiled_tolerance
            ):
                raise RuntimeError(
                    "New tiled evaluator differs "
                    "from model.predict_tiled(): "
                    f"ours={pred_tiled_count:.6f}, "
                    f"repo={repo_tiled_count:.6f}"
                )

        # ---------------------------------------------------------
        # Window bookkeeping + cancellation
        # ---------------------------------------------------------
        image_signed_errors: list[
            float
        ] = []

        for local_idx, row in enumerate(
            window_rows
        ):
            enriched = {
                "image_index": image_index,
                "window_index": local_idx,
                "image_path": sample[
                    "img_path"
                ],
                "image_height": H,
                "image_width": W,
                **row,
            }

            all_window_rows.append(
                enriched
            )
            image_signed_errors.append(
                float(
                    row["signed_error"]
                )
            )

        local_abs_sum = float(
            sum(
                abs(e)
                for e in image_signed_errors
            )
        )
        net_abs = abs(
            float(
                sum(
                    image_signed_errors
                )
            )
        )

        if local_abs_sum > 1e-12:
            cancel = (
                1.0
                - net_abs
                / local_abs_sum
            )
            cancel = float(
                min(
                    1.0,
                    max(
                        0.0,
                        cancel,
                    ),
                )
            )
        else:
            cancel = 0.0

        cancellation_ratios.append(
            cancel
        )

        sum_window_gt = float(
            sum(
                float(r["gt_count"])
                for r in window_rows
            )
        )
        sum_window_pred = float(
            sum(
                float(r["pred_count"])
                for r in window_rows
            )
        )

        if not math.isclose(
            sum_window_gt,
            gt_in_bounds,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                "Window GT partition mismatch."
            )

        if not math.isclose(
            sum_window_pred,
            pred_tiled_count,
            rel_tol=1e-5,
            abs_tol=1e-3,
        ):
            raise RuntimeError(
                "Window prediction partition "
                "mismatch."
            )

        # ---------------------------------------------------------
        # GAME
        # ---------------------------------------------------------
        direct_game = game_errors(
            y_direct,
            gt_y,
            args.game_levels,
        )
        tiled_game = game_errors(
            y_tiled,
            gt_y,
            args.game_levels,
        )

        for level in args.game_levels:
            level = int(level)
            direct_game_by_level[
                level
            ].append(
                direct_game[level]
            )
            tiled_game_by_level[
                level
            ].append(
                tiled_game[level]
            )

        # ---------------------------------------------------------
        # MICF validity / representation
        # ---------------------------------------------------------
        direct_valid = (
            measure_validity_metrics(
                y_direct
            )
        )
        tiled_valid = (
            measure_validity_metrics(
                y_tiled
            )
        )

        direct_repr = (
            representation_metrics(
                c_direct,
                y_direct,
                gt_c,
                gt_y,
                gt_in_bounds,
            )
        )
        tiled_repr = (
            representation_metrics(
                c_tiled,
                y_tiled,
                gt_c,
                gt_y,
                gt_in_bounds,
            )
        )

        direct_validity_rows.append(
            direct_valid
        )
        tiled_validity_rows.append(
            tiled_valid
        )
        direct_repr_rows.append(
            direct_repr
        )
        tiled_repr_rows.append(
            tiled_repr
        )

        # ---------------------------------------------------------
        # Per-image row
        # ---------------------------------------------------------
        row: dict[str, Any] = {
            "image_index": image_index,
            "image_path": sample[
                "img_path"
            ],
            "height": H,
            "width": W,
            "num_windows": len(
                window_rows
            ),
            "gt_count": gt_count,
            "gt_in_bounds_count": (
                gt_in_bounds
            ),
            "gt_out_of_bounds": (
                gt_out_of_bounds
            ),
            "pred_full_direct": (
                pred_direct_count
            ),
            "err_full_direct_signed": (
                pred_direct_count
                - gt_count
            ),
            "err_full_direct_abs": abs(
                pred_direct_count
                - gt_count
            ),
            "pred_full_tiled": (
                pred_tiled_count
            ),
            "err_full_tiled_signed": (
                pred_tiled_count
                - gt_count
            ),
            "err_full_tiled_abs": abs(
                pred_tiled_count
                - gt_count
            ),
            "cancellation_ratio": (
                cancel
            ),
            "window_abs_error_sum": (
                local_abs_sum
            ),
            "window_net_abs_error": (
                net_abs
            ),
            "sum_window_gt": (
                sum_window_gt
            ),
            "sum_window_predictions": (
                sum_window_pred
            ),
            "repo_tiled_count": (
                repo_tiled_count
            ),
            "repo_tiled_abs_diff": (
                repo_tiled_diff
            ),
        }

        for level, value in (
            direct_game.items()
        ):
            row[
                f"direct_game_L{level}"
            ] = value

        for level, value in (
            tiled_game.items()
        ):
            row[
                f"tiled_game_L{level}"
            ] = value

        for key, value in (
            direct_valid.items()
        ):
            row[
                f"direct_{key}"
            ] = value

        for key, value in (
            tiled_valid.items()
        ):
            row[
                f"tiled_{key}"
            ] = value

        for key, value in (
            direct_repr.items()
        ):
            row[
                f"direct_{key}"
            ] = value

        for key, value in (
            tiled_repr.items()
        ):
            row[
                f"tiled_{key}"
            ] = value

        per_image_rows.append(
            row
        )

        direct_predictions.append(
            pred_direct_count
        )
        tiled_predictions.append(
            pred_tiled_count
        )
        full_gt_counts.append(
            gt_count
        )

        print(
            f"[{image_index + 1:03d}/"
            f"{n_eval:03d}] "
            f"GT={gt_count:.1f} | "
            f"Direct={pred_direct_count:.2f} "
            f"(AE={abs(pred_direct_count-gt_count):.2f}) | "
            f"Tiled={pred_tiled_count:.2f} "
            f"(AE={abs(pred_tiled_count-gt_count):.2f}) | "
            f"Cancel={100*cancel:.1f}%"
        )

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------
    direct_stats = count_metric_summary(
        direct_predictions,
        full_gt_counts,
        eps=1.0,
    )
    tiled_stats = count_metric_summary(
        tiled_predictions,
        full_gt_counts,
        eps=1.0,
    )
    window_stats = window_metric_summary(
        all_window_rows
    )

    summary: dict[str, Any] = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "checkpoint_epoch": (
            checkpoint.get("epoch")
        ),
        "checkpoint_best_mae_stored": (
            checkpoint.get(
                "best_mae"
            )
        ),
        "model_head_type": (
            model.head_type
        ),
        "model_output_stride": (
            model.output_stride
        ),
        "model_finite_horizon": (
            model.finite_horizon
        ),
        "dataset_split": args.split,
        "num_images": n_eval,
        "num_windows": len(
            all_window_rows
        ),
        "tile_size": args.tile_size,
        "halo": args.halo,
        "game_levels": [
            int(x)
            for x in args.game_levels
        ],
        "window": window_stats,
        "full_tiled": tiled_stats,
        "full_direct": direct_stats,
        "direct_minus_tiled": {
            "mae_gap": (
                direct_stats["mae"]
                - tiled_stats["mae"]
            ),
            "rmse_gap": (
                direct_stats["rmse"]
                - tiled_stats["rmse"]
            ),
            "nae_gap": (
                direct_stats["nae"]
                - tiled_stats["nae"]
            ),
            "sre_gap": (
                direct_stats["sre"]
                - tiled_stats["sre"]
            ),
        },
        "cancellation": {
            "mean": finite_mean(
                cancellation_ratios
            ),
            "median": (
                finite_percentile(
                    cancellation_ratios,
                    50,
                )
            ),
            "p90": (
                finite_percentile(
                    cancellation_ratios,
                    90,
                )
            ),
        },
        "game_direct": {
            f"L{level}": finite_mean(
                values
            )
            for level, values
            in direct_game_by_level.items()
        },
        "game_tiled": {
            f"L{level}": finite_mean(
                values
            )
            for level, values
            in tiled_game_by_level.items()
        },
        "micf_validity_direct": {
            key: finite_mean(
                row[key]
                for row
                in direct_validity_rows
            )
            for key in (
                direct_validity_rows[0].keys()
                if direct_validity_rows
                else []
            )
        },
        "micf_validity_tiled": {
            key: finite_mean(
                row[key]
                for row
                in tiled_validity_rows
            )
            for key in (
                tiled_validity_rows[0].keys()
                if tiled_validity_rows
                else []
            )
        },
        "representation_direct": {
            key: finite_mean(
                row[key]
                for row
                in direct_repr_rows
            )
            for key in (
                direct_repr_rows[0].keys()
                if direct_repr_rows
                else []
            )
        },
        "representation_tiled": {
            key: finite_mean(
                row[key]
                for row
                in tiled_repr_rows
            )
            for key in (
                tiled_repr_rows[0].keys()
                if tiled_repr_rows
                else []
            )
        },
    }

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    summary_path = (
        output_dir
        / "comprehensive_summary.json"
    )
    image_csv = (
        output_dir
        / "comprehensive_per_image.csv"
    )
    window_csv = (
        output_dir
        / "comprehensive_per_window.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=True,
        )

    write_csv(
        image_csv,
        per_image_rows,
    )
    write_csv(
        window_csv,
        all_window_rows,
    )

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------
    print()
    print("=" * 96)
    print(
        "STANDARD COUNT METRICS"
    )
    print("=" * 96)
    print(
        f"{'Metric':<18}"
        f"{'Window':>14}"
        f"{'Tiled':>14}"
        f"{'Direct':>14}"
    )
    print("-" * 60)

    print(
        f"{'MAE/PMAE':<18}"
        f"{window_stats.get('mae', float('nan')):>14.4f}"
        f"{tiled_stats['mae']:>14.4f}"
        f"{direct_stats['mae']:>14.4f}"
    )
    print(
        f"{'RMSE/PRMSE':<18}"
        f"{window_stats.get('rmse', float('nan')):>14.4f}"
        f"{tiled_stats['rmse']:>14.4f}"
        f"{direct_stats['rmse']:>14.4f}"
    )
    print(
        f"{'NAE':<18}"
        f"{window_stats.get('nae_nonzero', float('nan')):>14.4f}"
        f"{tiled_stats['nae']:>14.4f}"
        f"{direct_stats['nae']:>14.4f}"
    )
    print(
        f"{'SRE':<18}"
        f"{'n/a':>14}"
        f"{tiled_stats['sre']:>14.4f}"
        f"{direct_stats['sre']:>14.4f}"
    )
    print(
        f"{'Signed Bias':<18}"
        f"{window_stats.get('signed_bias', float('nan')):>14.4f}"
        f"{tiled_stats['signed_bias']:>14.4f}"
        f"{direct_stats['signed_bias']:>14.4f}"
    )

    print()
    print("=" * 96)
    print(
        "TAIL / FAILURE METRICS"
    )
    print("=" * 96)

    for key, label in (
        ("median_ae", "Median AE"),
        ("p90_ae", "P90 AE"),
        ("p95_ae", "P95 AE"),
        ("max_ae", "Max AE"),
    ):
        print(
            f"{label:<18}"
            f"{window_stats.get(key, float('nan')):>14.4f}"
            f"{tiled_stats[key]:>14.4f}"
            f"{direct_stats[key]:>14.4f}"
        )

    print()
    print("=" * 96)
    print(
        "LOCAL / PATCH DIAGNOSTICS"
    )
    print("=" * 96)
    print(
        f"Window PMAE                 : "
        f"{window_stats.get('pmae', float('nan')):.4f}"
    )
    print(
        f"Window PRMSE                : "
        f"{window_stats.get('prmse', float('nan')):.4f}"
    )
    print(
        f"Window NAE (GT>0 only)      : "
        f"{window_stats.get('nae_nonzero', float('nan')):.4f}"
    )
    print(
        f"Non-empty Window MAE        : "
        f"{window_stats.get('nonempty_window_mae', float('nan')):.4f}"
    )
    print(
        f"Empty Window MAE            : "
        f"{window_stats.get('empty_window_mae', float('nan')):.4f}"
    )
    print(
        f"Empty Window Mean Pred      : "
        f"{window_stats.get('empty_window_mean_prediction', float('nan')):.4f}"
    )
    print(
        f"Empty Window Fraction       : "
        f"{100*window_stats.get('empty_window_fraction', float('nan')):.2f}%"
    )
    print(
        f"Mean Cancellation Ratio     : "
        f"{100*summary['cancellation']['mean']:.2f}%"
    )

    print()
    print("=" * 96)
    print(
        "GAME SPATIAL ERROR"
    )
    print("=" * 96)
    for level in args.game_levels:
        level = int(level)
        print(
            f"GAME({level}) "
            f"Tiled={summary['game_tiled'][f'L{level}']:.4f} "
            f"| Direct={summary['game_direct'][f'L{level}']:.4f}"
        )

    print()
    print("=" * 96)
    print(
        "MICF VALIDITY"
    )
    print("=" * 96)

    vd = summary[
        "micf_validity_direct"
    ]
    vt = summary[
        "micf_validity_tiled"
    ]

    for key in (
        "violation_rate",
        "violation_magnitude",
        "negative_mass_ratio",
        "negative_mass_total",
    ):
        print(
            f"{key:<28}"
            f"Tiled={vt.get(key, float('nan')):.6f} "
            f"| Direct={vd.get(key, float('nan')):.6f}"
        )

    print()
    print("=" * 96)
    print(
        "REPRESENTATION DIAGNOSTICS"
    )
    print("=" * 96)

    rd = summary[
        "representation_direct"
    ]
    rt = summary[
        "representation_tiled"
    ]

    for key in (
        "cumulative_field_nmae",
        "measure_nl1",
        "conservation_error",
    ):
        print(
            f"{key:<28}"
            f"Tiled={rt.get(key, float('nan')):.6f} "
            f"| Direct={rd.get(key, float('nan')):.6f}"
        )

    print()
    print("=" * 96)
    print(
        "DIRECT - TILED GAPS"
    )
    print("=" * 96)
    for key, value in (
        summary[
            "direct_minus_tiled"
        ].items()
    ):
        print(
            f"{key:<28}: "
            f"{value:.6f}"
        )

    print()
    print(f"Summary    : {summary_path}")
    print(f"Per-image  : {image_csv}")
    print(f"Per-window : {window_csv}")


if __name__ == "__main__":
    main()
```

---

# 14. Run commands

## B5b

```bash
python tools/eval_micf_comprehensive.py \
  --checkpoint runs/pilot_micf/b5b/best.pt \
  --device cuda \
  --tile-size 256 \
  --halo 64 \
  --game-levels 0 1 2 3 \
  --verify-repo-tiled
```

## B8 K=4

```bash
python tools/eval_micf_comprehensive.py \
  --checkpoint runs/pilot_micf/b8_k4/best.pt \
  --device cuda \
  --tile-size 256 \
  --halo 64 \
  --game-levels 0 1 2 3 \
  --verify-repo-tiled
```

## Debug first five images

```bash
python tools/eval_micf_comprehensive.py \
  --checkpoint runs/pilot_micf/b5b/best.pt \
  --device cuda \
  --tile-size 256 \
  --halo 64 \
  --game-levels 0 1 2 \
  --max-samples 5 \
  --verify-repo-tiled
```

---

# 15. Output files

```text
runs/pilot_micf/<model>/eval_comprehensive/
├── comprehensive_summary.json
├── comprehensive_per_image.csv
└── comprehensive_per_window.csv
```

The JSON contains all aggregated metrics.

The per-image CSV contains:

```text
GT / Direct / Tiled predictions
signed and absolute errors
GAME(0..L)
MICF validity
representation diagnostics
cancellation
repo-tiled consistency
```

The per-window CSV contains:

```text
window coordinates
GT count
predicted count
signed error
absolute error
```

---

# 16. Recommended result tables

## 16.1 Main count table

| Model | Window PMAE ↓ | Window PRMSE ↓ | Tiled MAE ↓ | Tiled RMSE ↓ | Tiled NAE ↓ | Direct MAE ↓ | Direct RMSE ↓ | Direct NAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B5b | | | | | | | | |
| B8 K=4 | | | | | | | | |

---

## 16.2 Spatial / failure table

| Model | GAME(1) ↓ | GAME(2) ↓ | GAME(3) ↓ | P95 AE Direct ↓ | Max AE Direct ↓ | Cancellation ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B5b | | | | | | |
| B8 K=4 | | | | | | |

---

## 16.3 MICF representation table

| Model | Direct Viol.% ↓ | Direct NegMass ↓ | Tiled Viol.% ↓ | Tiled NegMass ↓ | Direct C-NMAE ↓ | Tiled C-NMAE ↓ | Direct Y-NL1 ↓ | Tiled Y-NL1 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B5b | | | | | | | | |
| B8 K=4 | | | | | | | | |

---

## 16.4 Global-horizon diagnostic table

| Model | MAE Direct−Tiled ↓ | RMSE Direct−Tiled ↓ | NAE Direct−Tiled ↓ | SRE Direct−Tiled ↓ |
|---|---:|---:|---:|---:|
| B5b | | | | |
| B8 K=4 | | | | |

---

# 17. Interpretation

A strong finite-horizon signal is not merely:

\[
MAE_{B8}<MAE_{B5b}.
\]

A much stronger pattern is:

\[
PMAE_{B8}
\lesssim
PMAE_{B5b},
\]

\[
GAME_{B8}
\lesssim
GAME_{B5b},
\]

\[
NMR_{B8}
\lesssim
NMR_{B5b},
\]

and:

\[
MAE_{direct,B8}
<
MAE_{direct,B5b}
\]

with a reduced:

\[
MAE_{direct}-MAE_{tiled}.
\]

That would indicate that B8 does not merely improve one scalar count; it improves the cumulative representation without sacrificing local/spatial measure quality.

Conversely:

- low full-image MAE + high GAME / high \(Y\)-NL1 → spatial errors are cancelling;
- low PMAE + high Direct−Tiled gap → local counting works but direct global extrapolation is weak;
- high negative-mass ratio → count may be numerically good while the MICF measure is invalid;
- high P95/Max AE with decent MAE → rare catastrophic failures remain.

---

# 18. Final metric checklist

The evaluator must not be considered complete unless it outputs all of the following:

### Standard counting

- [x] MAE
- [x] RMSE / crowd-counting “MSE” convention
- [x] NAE
- [x] SRE

### Local counting

- [x] Window PMAE
- [x] Window PRMSE
- [x] non-zero-window NAE
- [x] non-empty-window MAE
- [x] empty-window MAE
- [x] empty-window mean prediction
- [x] GAME(0–3)

### Error distribution

- [x] signed bias
- [x] median AE
- [x] P90 AE
- [x] P95 AE
- [x] max AE

### MICF validity

- [x] violation rate
- [x] violation magnitude
- [x] negative-mass ratio
- [x] total negative mass
- [x] count↔measure conservation error

### Representation diagnostics

- [x] cumulative-field NMAE
- [x] local-measure normalized L1
- [x] cancellation ratio
- [x] Direct−Tiled MAE gap
- [x] Direct−Tiled RMSE gap
- [x] Direct−Tiled NAE gap
- [x] Direct−Tiled SRE gap

This suite is intentionally comprehensive enough to diagnose **where** MICF/B5b/B8 fails, instead of reducing evaluation to one MAE number.
