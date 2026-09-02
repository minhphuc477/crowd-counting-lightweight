# E0-v2 Compression Attribution Audit
## Full implementation specification for `crowd-counting-lightweight`

**Repository:** `minhphuc477/crowd-counting-lightweight`  
**Branch:** `feat/ntpc-neural-tree-polya`  
**Audited branch HEAD:** `753ceb180ee4f1e20cd00924b44013e7479e7398`  
**Purpose:** test whether the current ultra-light crowd-counting backbone loses **locally decodable cardinality information** at `C4→C8` and/or `C8→C16`, and separate this from a simple representation-dimension/channel-budget effect.

---

# 0. Decision before touching the architecture

Do **not** modify `HPCLite`, `AdditiveFPNNeck`, NTPC losses, the density/mass head, or the production training loop yet.

The repository already provides the two things required by E0-v2:

1. Backbone features at reductions `C4`, `C8`, `C16` through `MobileNetV4Backbone`.
2. Exact integer point-derived targets `Y4 → Y8 → Y16 → Y32 → Y64`, with exact count conservation.

Therefore E0-v2 should be a **diagnostic-only addition**. If E0-v2 fails, the candidate research direction is killed without contaminating the deployed graph.

The current research question is not:

> “Does downsampling lose information?”

That is too generic and already occupied.

The precise question is:

> **At a fixed ultra-light compute budget, does a native spatial-compression transition lose more locally decodable cardinality information than a lossless rearrangement followed by a matched linear channel-budget control?**

For a transition `s → 2s`, define the exact local target

\[
Y_R = [n_{TL}, n_{TR}, n_{BL}, n_{BR}].
\]

Let the pre-transition feature be \(X_s\), the native post-transition feature be \(Z_{2s}\), and the lossless packed pre-feature be

\[
U_{2s} = \operatorname{SpaceToDepth}_2(X_s).
\]

`SpaceToDepth` is only a re-indexing operation, so it does not discard values.  
We then compare:

- **`pre_pack`**: \(U_{2s}\), an information-preserving pre-transition reference;
- **`native_post`**: actual backbone output \(Z_{2s}\);
- **`s2d_pca_budget`**: `pre_pack` compressed by train-only PCA to the **same output channel dimension** as `native_post`;
- **`avgpool_pca_budget`**: ordinary average decimation, then PCA to the native channel budget;
- **`blurpool_pca_budget`**: fixed anti-aliased decimation, then PCA to the native channel budget.

PCA here is **not claimed to be a theoretical upper bound**. It is an unsupervised, budget-matched linear compression control.

The primary diagnostic quantity is held-out child-count prediction error using the same probe family:

\[
\Delta_{native-pre}
=
R_{\mathcal H}(Z_{2s})
-
R_{\mathcal H}(U_{2s}),
\]

and the more important budget comparison is

\[
\Delta_{excess}
=
R_{\mathcal H}(Z_{2s})
-
R_{\mathcal H}(P_{k}U_{2s}),
\]

where \(k=C_{post}\) and \(P_k\) is fitted **only on the diagnostic training partition**.

---

# 1. Files to add

Add exactly these files:

```text
hpc/diagnostics/cardinality_sufficiency_v2.py
tools/run_cardinality_sufficiency_v2.py
tools/summarize_cardinality_sufficiency_v2.py
tests/test_cardinality_sufficiency_v2.py
```

No production model file needs to change for E0-v2.

Optional: export the functions from `hpc/diagnostics/__init__.py`. This is not required because the runner imports the module directly.

---

# 2. Full code — `hpc/diagnostics/cardinality_sufficiency_v2.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F


BIN_NAMES = ("n1", "n2", "n3_4", "n5p")


def pack_2x2_features(x: torch.Tensor) -> torch.Tensor:
    """Losslessly pack each non-overlapping 2x2 feature neighborhood.

    Input:  [B, C, H, W]
    Output: [B, H/2, W/2, 4C] in TL, TR, BL, BR order.

    This is equivalent to a feature-space SpaceToDepth(2) rearrangement and is
    information-preserving up to tensor reordering.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    _, _, h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"H/W must be even for 2x2 packing, got {(h, w)}")
    tl = x[:, :, 0::2, 0::2]
    tr = x[:, :, 0::2, 1::2]
    bl = x[:, :, 1::2, 0::2]
    br = x[:, :, 1::2, 1::2]
    return torch.cat((tl, tr, bl, br), dim=1).permute(0, 2, 3, 1).contiguous()


def post_to_cell_vectors(x: torch.Tensor) -> torch.Tensor:
    """Convert post-transition map [B,C,H,W] to [B,H,W,C]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


def pack_child_counts(y: torch.Tensor) -> torch.Tensor:
    """Pack exact four-child counts for a stride-s -> stride-2s transition.

    Input:  [B,H,W] or [B,1,H,W] exact integer count grid at stride s.
    Output: [B,H/2,W/2,4] in TL, TR, BL, BR order.
    """
    if y.ndim == 4:
        if y.shape[1] != 1:
            raise ValueError(f"4D count map must have one channel, got {tuple(y.shape)}")
        y = y[:, 0]
    if y.ndim != 3:
        raise ValueError(f"Expected [B,H,W] or [B,1,H,W], got {tuple(y.shape)}")
    _, h, w = y.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"H/W must be even for child packing, got {(h, w)}")
    return torch.stack(
        (
            y[:, 0::2, 0::2],
            y[:, 0::2, 1::2],
            y[:, 1::2, 0::2],
            y[:, 1::2, 1::2],
        ),
        dim=-1,
    ).contiguous()


def blurpool2x(x: torch.Tensor) -> torch.Tensor:
    """Fixed anti-aliased 2x decimation control using [1,2,1]^2 / 16.

    This is a diagnostic control, not a proposed architecture. Replicate padding
    is used to keep an even HxW tensor aligned to exactly H/2 x W/2 cells.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    _, c, h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"H/W must be even, got {(h, w)}")
    k1 = x.new_tensor([1.0, 2.0, 1.0])
    kernel = torch.outer(k1, k1)
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(x_pad, kernel, stride=2, groups=c)


def avgpool2x(x: torch.Tensor) -> torch.Tensor:
    """Plain 2x average-pooling diagnostic control."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def parent_count_bin(n: torch.Tensor) -> torch.Tensor:
    """Map parent counts to 0:n1, 1:n2, 2:n3-4, 3:n5+."""
    n = n.float()
    out = torch.full_like(n, -1, dtype=torch.long)
    out[n == 1] = 0
    out[n == 2] = 1
    out[(n >= 3) & (n <= 4)] = 2
    out[n >= 5] = 3
    return out


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-6) -> "Standardizer":
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("Need [N,D] with N>=2")
        xd = x.double()
        return cls(
            mean=xd.mean(dim=0),
            scale=xd.std(dim=0, unbiased=False).clamp_min(eps),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.double() - self.mean) / self.scale


@dataclass
class PCAProjector:
    mean: torch.Tensor
    basis: torch.Tensor
    output_dim: int

    @classmethod
    def fit(cls, x: torch.Tensor, output_dim: int) -> "PCAProjector":
        """Fit unsupervised PCA on a [N,D] matrix using a D x D covariance eigendecomposition.

        If output_dim > D, all D components are retained and zeros are appended at
        transform time. Zero padding adds no information and keeps the requested
        interface dimension explicit.
        """
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("Need [N,D] with N>=2")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        xd = x.double()
        mean = xd.mean(dim=0)
        xc = xd - mean
        denom = max(int(xc.shape[0]) - 1, 1)
        cov = (xc.T @ xc) / float(denom)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        k = min(int(output_dim), int(x.shape[1]))
        order = torch.argsort(eigvals, descending=True)[:k]
        basis = eigvecs[:, order].contiguous()
        return cls(mean=mean, basis=basis, output_dim=int(output_dim))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.double() - self.mean) @ self.basis
        if z.shape[1] < self.output_dim:
            pad = z.new_zeros((z.shape[0], self.output_dim - z.shape[1]))
            z = torch.cat((z, pad), dim=1)
        return z.float()


def fit_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, Standardizer]:
    """Closed-form multi-output ridge with an unpenalized intercept."""
    if x.ndim != 2:
        raise ValueError("x must be [N,D]")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("y must be [N,O] aligned with x")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    standardizer = Standardizer.fit(x)
    xs = standardizer.transform(x)
    ones = torch.ones((xs.shape[0], 1), dtype=torch.float64, device=xs.device)
    xa = torch.cat((xs, ones), dim=1)
    lhs = xa.T @ xa
    reg = torch.eye(lhs.shape[0], dtype=torch.float64, device=lhs.device) * float(alpha)
    reg[-1, -1] = 0.0
    rhs = xa.T @ y.double()
    try:
        weights = torch.linalg.solve(lhs + reg, rhs)
    except RuntimeError:
        weights = torch.linalg.pinv(lhs + reg) @ rhs
    return weights, standardizer


def predict_ridge(
    x: torch.Tensor,
    weights: torch.Tensor,
    standardizer: Standardizer,
) -> torch.Tensor:
    xs = standardizer.transform(x)
    ones = torch.ones((xs.shape[0], 1), dtype=torch.float64, device=xs.device)
    xa = torch.cat((xs, ones), dim=1)
    return (xa @ weights).float()


def fit_predict_ridge(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    weights, standardizer = fit_ridge(x_train, y_train, alpha=alpha)
    return predict_ridge(x_val, weights, standardizer)


def per_cell_child_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    return (pred.float() - target.float()).abs().mean(dim=1)


def per_cell_parent_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    return (pred.float().sum(dim=1) - target.float().sum(dim=1)).abs()


def per_cell_composition_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L1 error between normalized four-child compositions, active parents only."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    p = pred.float().clamp_min(0.0)
    t = target.float()
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    return (p - t).abs().sum(dim=1)


def summarize_prediction(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    child = per_cell_child_mae(pred, target)
    parent = per_cell_parent_mae(pred, target)
    composition = per_cell_composition_l1(pred, target)
    result: Dict[str, float] = {
        "cells": int(target.shape[0]),
        "child_mae": float(child.mean()) if child.numel() else float("nan"),
        "parent_mae": float(parent.mean()) if parent.numel() else float("nan"),
        "composition_l1": float(composition.mean()) if composition.numel() else float("nan"),
        "negative_prediction_fraction": float((pred < 0).float().mean()) if pred.numel() else float("nan"),
    }
    n = target.sum(dim=1)
    labels = parent_count_bin(n)
    for idx, name in enumerate(BIN_NAMES):
        mask = labels == idx
        result[f"{name}_cells"] = int(mask.sum())
        result[f"{name}_child_mae"] = float(child[mask].mean()) if mask.any() else float("nan")
        result[f"{name}_parent_mae"] = float(parent[mask].mean()) if mask.any() else float("nan")
        result[f"{name}_composition_l1"] = float(composition[mask].mean()) if mask.any() else float("nan")
    multi = n >= 2
    result["n2p_cells"] = int(multi.sum())
    result["n2p_child_mae"] = float(child[multi].mean()) if multi.any() else float("nan")
    result["n2p_parent_mae"] = float(parent[multi].mean()) if multi.any() else float("nan")
    result["n2p_composition_l1"] = float(composition[multi].mean()) if multi.any() else float("nan")
    return result


def bootstrap_image_mean_difference(
    a: torch.Tensor,
    b: torch.Tensor,
    image_ids: torch.Tensor,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Image-level bootstrap CI for mean(a-b), avoiding cell-level pseudo-replication."""
    if a.ndim != 1 or b.ndim != 1 or image_ids.ndim != 1:
        raise ValueError("a, b, image_ids must be 1D")
    if not (a.numel() == b.numel() == image_ids.numel()):
        raise ValueError("a, b, image_ids must have equal length")
    unique = torch.unique(image_ids.cpu())
    if unique.numel() < 2:
        return {"mean_diff": float((a - b).mean()), "ci95_low": float("nan"), "ci95_high": float("nan"), "images": int(unique.numel())}

    per_image = []
    for image_id in unique.tolist():
        mask = image_ids.cpu() == int(image_id)
        per_image.append(float((a.cpu()[mask] - b.cpu()[mask]).mean()))
    values = torch.tensor(per_image, dtype=torch.float64)
    g = torch.Generator().manual_seed(int(seed))
    boot = torch.empty(int(n_boot), dtype=torch.float64)
    m = values.numel()
    for i in range(int(n_boot)):
        idx = torch.randint(0, m, (m,), generator=g)
        boot[i] = values[idx].mean()
    q = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "mean_diff": float(values.mean()),
        "ci95_low": float(q[0]),
        "ci95_high": float(q[1]),
        "images": int(m),
    }


def build_representation_grid(
    pre: torch.Tensor,
    post: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return aligned diagnostic representation grids before flattening.

    Keys:
      pre_pack: exact 2x2 packing, [B,H/2,W/2,4C_pre]
      native_post: native backbone output, [B,H/2,W/2,C_post]
      avgpool: plain average-decimated pre feature, [B,H/2,W/2,C_pre]
      blurpool: anti-aliased decimated pre feature, [B,H/2,W/2,C_pre]
    """
    pre_pack = pack_2x2_features(pre)
    native_post = post_to_cell_vectors(post)
    avg = post_to_cell_vectors(avgpool2x(pre))
    blur = post_to_cell_vectors(blurpool2x(pre))
    expected = pre_pack.shape[:3]
    for name, value in {"native_post": native_post, "avgpool": avg, "blurpool": blur}.items():
        if value.shape[:3] != expected:
            raise ValueError(
                f"Grid mismatch for {name}: expected {tuple(expected)}, got {tuple(value.shape[:3])}"
            )
    return {
        "pre_pack": pre_pack,
        "native_post": native_post,
        "avgpool": avg,
        "blurpool": blur,
    }


def flatten_selected_cells(
    grid: torch.Tensor,
    target_children: torch.Tensor,
    image_id: int,
    min_parent_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten aligned representation/target grids and keep active parents."""
    if grid.ndim != 4 or target_children.ndim != 4 or target_children.shape[-1] != 4:
        raise ValueError("Expected grid [B,H,W,D] and targets [B,H,W,4]")
    if grid.shape[:3] != target_children.shape[:3]:
        raise ValueError("Representation and target grids are not aligned")
    if grid.shape[0] != 1:
        raise ValueError("Diagnostic collector expects batch size 1 for stable crop metadata")
    x = grid.reshape(-1, grid.shape[-1]).float()
    y = target_children.reshape(-1, 4).float()
    n = y.sum(dim=1)
    mask = n >= float(min_parent_count)
    ids = torch.full((int(mask.sum()),), int(image_id), dtype=torch.long)
    return x[mask], y[mask], n[mask], ids


class TinyMLPProbe(torch.nn.Module):
    """Fixed-capacity nonlinear accessibility probe; diagnostic only."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_predict_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    *,
    hidden: int = 64,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Train a fixed-protocol tiny MLP and return held-out predictions on CPU.

    No validation-driven early stopping or hyperparameter selection is used; every
    representation receives the same pre-registered protocol. The input layer
    necessarily scales with representation dimensionality, so budget-matched
    comparisons should focus on native_post vs PCA-budget controls.
    """
    if hidden <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("hidden, epochs and batch_size must be positive")
    standardizer = Standardizer.fit(x_train)
    xtr = standardizer.transform(x_train).float()
    xva = standardizer.transform(x_val).float()
    ytr = y_train.float()
    dev = torch.device(device)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = TinyMLPProbe(xtr.shape[1], hidden=hidden).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    generator = torch.Generator().manual_seed(int(seed))
    model.train()
    n = xtr.shape[0]
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            xb = xtr[idx].to(dev)
            yb = ytr[idx].to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=1.0)
            loss.backward()
            optimizer.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, xva.shape[0], int(batch_size)):
            preds.append(model(xva[start : start + int(batch_size)].to(dev)).cpu())
    return torch.cat(preds, dim=0)

```

---

# 3. Full code — `tools/run_cardinality_sufficiency_v2.py`

```python
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr

from hpc.data.nwpu import NWPUDataset
from hpc.data.point_counts import block_sum
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.sha import ShanghaiTechDataset
from hpc.models.factory import (
    assert_checkpoint_compatible,
    build_model_from_config,
    validate_pretrained_normalization,
)

from hpc.diagnostics.cardinality_sufficiency_v2 import (
    BIN_NAMES,
    PCAProjector,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_mlp,
    fit_predict_ridge,
    pack_child_counts,
    parent_count_bin,
    per_cell_child_mae,
    summarize_prediction,
)


TRANSITIONS = ((4, 8), (8, 16))
REP_BASE = ("pre_pack", "native_post", "avgpool", "blurpool")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _stable_seed(base: int, *parts: object) -> int:
    payload = ":".join([str(base), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


@contextlib.contextmanager
def temporary_rng(seed: int):
    """Make dataset random crop/scale/flip deterministic without leaking RNG changes."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def split_image_indices(
    image_paths: Sequence[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction must be in [0.05, 0.5]")
    indices = list(range(len(image_paths)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * float(val_fraction))))
    n_val = min(n_val, len(indices) - 1)
    return indices[n_val:], indices[:n_val]


def resolve_dataset_normalization(cfg: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ds = cfg.get("dataset", {})
    mean = tuple(float(v) for v in ds.get("image_mean", [0.485, 0.456, 0.406]))
    std = tuple(float(v) for v in ds.get("image_std", [0.229, 0.224, 0.225]))
    return mean, std


def build_train_dataset(
    cfg: dict,
    *,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
):
    ds = cfg["dataset"]
    aug = cfg.get("augmentation", {})
    name = str(ds.get("name", "sha")).lower().replace("-", "_")
    mean = image_mean if image_mean is not None else ds.get("image_mean", [0.485, 0.456, 0.406])
    std = image_std if image_std is not None else ds.get("image_std", [0.229, 0.224, 0.225])
    common = dict(
        crop_size=int(ds.get("crop_size", 256)),
        is_train=True,
        scale_range=tuple(float(v) for v in aug.get("scale_range", [0.7, 1.3])),
        flip_prob=float(aug.get("flip_prob", 0.5)),
        image_mean=mean,
        image_std=std,
    )
    if "coordinate_base" in ds:
        common["coordinate_base"] = int(ds["coordinate_base"])

    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = ds.get("part", "part_B" if name.endswith("_b") else "part_A")
        return ShanghaiTechDataset(root=ds["root"], part=part, split="train_data", **common)
    if name in {"qnrf", "ucf_qnrf"}:
        return UCFQNRFDataset(root=ds["root"], split="Train", **common)
    if name == "nwpu":
        return NWPUDataset(
            root=ds["root"],
            split="train",
            split_file=ds.get("train_split_file"),
            **common,
        )
    raise ValueError(f"Unsupported dataset for E0-v2: {name}")


class GenericTimmReductionExtractor(nn.Module):
    """Diagnostic-only pretrained feature extractor for cross-backbone controls.

    Unlike the production MobileNetV4Backbone, this wrapper does not physically
    truncate a model and therefore works with timm backbones whose stage names
    are not `blocks.*`. It must not replace the production model silently.
    """

    def __init__(self, model_name: str, reductions: Sequence[int] = (4, 8, 16)):
        super().__init__()
        import timm

        with torch.random.fork_rng(devices=[]):
            probe = timm.create_model(model_name, pretrained=False, features_only=True)
            all_reductions = list(probe.feature_info.reduction())
            del probe
        selected = []
        for reduction in reductions:
            matches = [i for i, r in enumerate(all_reductions) if int(r) == int(reduction)]
            if not matches:
                raise ValueError(
                    f"Reduction {reduction} unavailable for {model_name}; got {all_reductions}"
                )
            selected.append(matches[-1])
        self.reductions = tuple(int(v) for v in reductions)
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=tuple(selected),
        )

    def forward(self, x: torch.Tensor):
        return tuple(self.model(x))


def resolve_timm_normalization(model_name: str) -> tuple[tuple[float, ...], tuple[float, ...], dict]:
    import timm

    pretrained_cfg = timm.get_pretrained_cfg(model_name)
    if pretrained_cfg is None:
        raise ValueError(f"No timm pretrained config for {model_name}")
    source = pretrained_cfg.hf_hub_id or pretrained_cfg.url or pretrained_cfg.file
    if not source:
        raise ValueError(f"No pretrained weights source for {model_name}")
    meta = {
        "architecture": pretrained_cfg.architecture,
        "tag": pretrained_cfg.tag,
        "source": str(source),
        "mean": list(map(float, pretrained_cfg.mean)),
        "std": list(map(float, pretrained_cfg.std)),
    }
    return tuple(meta["mean"]), tuple(meta["std"]), meta


@dataclass
class LoadedSource:
    extractor: nn.Module
    full_model: nn.Module | None
    reductions: tuple[int, ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    provenance: dict


def load_feature_source(
    cfg: dict,
    device: torch.device,
    checkpoint: str | None,
    backbone_override: str | None,
) -> LoadedSource:
    if checkpoint and backbone_override:
        raise ValueError("Use either --checkpoint or --backbone-override, not both")

    if backbone_override:
        mean, std, meta = resolve_timm_normalization(backbone_override)
        extractor = GenericTimmReductionExtractor(backbone_override, reductions=(4, 8, 16)).to(device).eval()
        for p in extractor.parameters():
            p.requires_grad_(False)
        return LoadedSource(
            extractor=extractor,
            full_model=None,
            reductions=(4, 8, 16),
            normalization_mean=mean,
            normalization_std=std,
            provenance={"kind": "generic_timm_pretrained", "model": backbone_override, **meta},
        )

    # Current production carrier: config normalization must match its pretrained weights.
    pretrained_spec = validate_pretrained_normalization(cfg)
    mean, std = resolve_dataset_normalization(cfg)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert_checkpoint_compatible(ckpt, cfg)
        model = build_model_from_config(cfg, load_pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        provenance = {
            "kind": "task_checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_best_mae": float(ckpt.get("best_mae", float("nan"))),
            "checkpoint_git_sha": ckpt.get("runtime", {}).get("git_sha"),
        }
    else:
        model = build_model_from_config(cfg, load_pretrained=True)
        provenance = {
            "kind": "current_timm_pretrained",
            "pretrained_spec": pretrained_spec,
        }
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    reductions = tuple(int(v) for v in model.backbone.target_reductions)
    return LoadedSource(
        extractor=model.backbone,
        full_model=model if checkpoint else None,
        reductions=reductions,
        normalization_mean=mean,
        normalization_std=std,
        provenance=provenance,
    )


class TransitionReservoir:
    """Deterministic stratified active-cell reservoir with aligned representations."""

    def __init__(self, max_per_bin: int, seed: int):
        if max_per_bin <= 0:
            raise ValueError("max_per_bin must be positive")
        self.max_per_bin = int(max_per_bin)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.counts = {name: 0 for name in BIN_NAMES}
        self.rep_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {rep: [] for rep in REP_BASE} for name in BIN_NAMES
        }
        self.y_chunks = {name: [] for name in BIN_NAMES}
        self.id_chunks = {name: [] for name in BIN_NAMES}
        self.downstream_chunks = {name: [] for name in BIN_NAMES}

    def add(
        self,
        reps: dict[str, torch.Tensor],
        y_children: torch.Tensor,
        image_id: int,
        downstream_parent_pred: torch.Tensor | None,
    ) -> None:
        if y_children.ndim != 4 or y_children.shape[0] != 1 or y_children.shape[-1] != 4:
            raise ValueError("Expected y_children [1,H,W,4]")
        y = y_children.reshape(-1, 4).float().cpu()
        n = y.sum(dim=1)
        labels = parent_count_bin(n)
        flat_reps = {
            name: tensor.reshape(-1, tensor.shape[-1]).float().cpu()
            for name, tensor in reps.items()
        }
        for name, tensor in flat_reps.items():
            if tensor.shape[0] != y.shape[0]:
                raise ValueError(f"Representation {name} is not target-aligned")

        if downstream_parent_pred is not None:
            downstream = downstream_parent_pred.reshape(-1).float().cpu()
            if downstream.numel() != y.shape[0]:
                raise ValueError("downstream_parent_pred is not target-aligned")
        else:
            downstream = torch.full((y.shape[0],), float("nan"))

        for bin_idx, bin_name in enumerate(BIN_NAMES):
            remaining = self.max_per_bin - self.counts[bin_name]
            if remaining <= 0:
                continue
            idx = torch.nonzero(labels == bin_idx, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            order = torch.randperm(idx.numel(), generator=self.generator)
            idx = idx[order[:remaining]]
            if idx.numel() == 0:
                continue
            for rep_name in REP_BASE:
                self.rep_chunks[bin_name][rep_name].append(flat_reps[rep_name][idx])
            self.y_chunks[bin_name].append(y[idx])
            self.id_chunks[bin_name].append(
                torch.full((idx.numel(),), int(image_id), dtype=torch.long)
            )
            self.downstream_chunks[bin_name].append(downstream[idx])
            self.counts[bin_name] += int(idx.numel())

    def finalize(self) -> dict:
        reps: dict[str, list[torch.Tensor]] = {rep: [] for rep in REP_BASE}
        y_all, ids_all, downstream_all = [], [], []
        bin_slices: dict[str, list[int]] = {}
        cursor = 0
        for bin_name in BIN_NAMES:
            if not self.y_chunks[bin_name]:
                bin_slices[bin_name] = [cursor, cursor]
                continue
            y = torch.cat(self.y_chunks[bin_name], dim=0)
            ids = torch.cat(self.id_chunks[bin_name], dim=0)
            downstream = torch.cat(self.downstream_chunks[bin_name], dim=0)
            for rep in REP_BASE:
                reps[rep].append(torch.cat(self.rep_chunks[bin_name][rep], dim=0))
            y_all.append(y)
            ids_all.append(ids)
            downstream_all.append(downstream)
            start = cursor
            cursor += y.shape[0]
            bin_slices[bin_name] = [start, cursor]
        if not y_all:
            raise RuntimeError("No active cells were collected")
        return {
            "reps": {rep: torch.cat(chunks, dim=0) for rep, chunks in reps.items()},
            "y": torch.cat(y_all, dim=0),
            "image_ids": torch.cat(ids_all, dim=0),
            "downstream_parent_pred": torch.cat(downstream_all, dim=0),
            "bin_slices": bin_slices,
            "counts": dict(self.counts),
        }


def collect_split(
    dataset,
    indices: Sequence[int],
    source: LoadedSource,
    device: torch.device,
    crops_per_image: int,
    max_cells_per_bin: int,
    seed: int,
) -> dict[tuple[int, int], dict]:
    reservoirs = {
        transition: TransitionReservoir(
            max_per_bin=max_cells_per_bin,
            seed=_stable_seed(seed, "reservoir", *transition),
        )
        for transition in TRANSITIONS
        if transition[0] in source.reductions and transition[1] in source.reductions
    }
    if not reservoirs:
        raise ValueError(f"No supported transitions in reductions={source.reductions}")

    shuffled = list(indices)
    random.Random(_stable_seed(seed, "image_order")).shuffle(shuffled)

    with torch.no_grad():
        for position, image_idx in enumerate(shuffled):
            for crop_id in range(int(crops_per_image)):
                crop_seed = _stable_seed(seed, "crop", image_idx, crop_id)
                with temporary_rng(crop_seed):
                    sample = dataset[image_idx]
                image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
                features = source.extractor(image)
                feature_by_stride = {
                    int(stride): feat.float()
                    for stride, feat in zip(source.reductions, features)
                }

                mass = None
                if source.full_model is not None:
                    mass = source.full_model(image).float()
                    if mass.shape[-2:] != sample["gt_blocks"][4].shape[-2:]:
                        raise RuntimeError(
                            "Task mass map and exact Y4 are not aligned: "
                            f"mass={tuple(mass.shape)}, y4={tuple(sample['gt_blocks'][4].shape)}"
                        )

                for (s, t), reservoir in reservoirs.items():
                    pre = feature_by_stride[s]
                    post = feature_by_stride[t]
                    reps = build_representation_grid(pre, post)
                    y_s = sample["gt_blocks"][s].unsqueeze(0).to(device)
                    y_children = pack_child_counts(y_s).cpu()
                    if reps["native_post"].shape[:3] != y_children.shape[:3]:
                        raise RuntimeError(
                            f"Feature/target geometry mismatch at {s}->{t}: "
                            f"post={tuple(reps['native_post'].shape)}, target={tuple(y_children.shape)}"
                        )

                    downstream_parent_pred = None
                    if mass is not None:
                        factor = t // 4
                        if t % 4 != 0:
                            raise ValueError(f"Transition parent stride {t} not divisible by output stride 4")
                        downstream_parent_pred = block_sum(mass[:, 0], factor).cpu()
                        y_parent = sample["gt_blocks"][t].unsqueeze(0)
                        if downstream_parent_pred.shape != y_parent.shape:
                            raise RuntimeError(
                                f"Downstream local count mismatch at stride {t}: "
                                f"pred={tuple(downstream_parent_pred.shape)}, gt={tuple(y_parent.shape)}"
                            )

                    reservoir.add(
                        {name: tensor.cpu() for name, tensor in reps.items()},
                        y_children,
                        image_id=image_idx,
                        downstream_parent_pred=downstream_parent_pred,
                    )

            if (position + 1) % 20 == 0:
                counts_text = ", ".join(
                    f"{s}->{t}:{reservoir.counts}"
                    for (s, t), reservoir in reservoirs.items()
                )
                print(f"Collected {position + 1}/{len(shuffled)} images | {counts_text}", flush=True)

    return {transition: reservoir.finalize() for transition, reservoir in reservoirs.items()}


def fit_budget_representations(train: dict, val: dict) -> tuple[dict, dict, dict]:
    """Fit train-only PCA controls to the native post-transition channel budget."""
    train_reps = dict(train["reps"])
    val_reps = dict(val["reps"])
    post_dim = int(train_reps["native_post"].shape[1])
    meta = {"native_post_dim": post_dim, "input_dims": {k: int(v.shape[1]) for k, v in train_reps.items()}}

    for source_name, target_name in (
        ("pre_pack", "s2d_pca_budget"),
        ("avgpool", "avgpool_pca_budget"),
        ("blurpool", "blurpool_pca_budget"),
    ):
        projector = PCAProjector.fit(train_reps[source_name], output_dim=post_dim)
        train_reps[target_name] = projector.transform(train_reps[source_name])
        val_reps[target_name] = projector.transform(val_reps[source_name])
        meta[target_name] = {
            "source_dim": int(train_reps[source_name].shape[1]),
            "output_dim": post_dim,
            "effective_rank_cap": min(int(train_reps[source_name].shape[1]), post_dim),
        }
    return train_reps, val_reps, meta


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-12 else float("nan")


def evaluate_transition(
    train: dict,
    val: dict,
    ridge_alpha: float,
    bootstrap: int,
    seed: int,
    min_relative_degradation: float,
    min_linkage_rho: float,
    run_mlp: bool = False,
    mlp_epochs: int = 40,
    probe_device: str = "cpu",
) -> dict:
    train_reps, val_reps, projection_meta = fit_budget_representations(train, val)
    y_train = train["y"]
    y_val = val["y"]
    image_ids = val["image_ids"]
    n_val = y_val.sum(dim=1)
    multi = n_val >= 2

    predictions = {}
    metrics = {}
    for rep_name in (
        "pre_pack",
        "native_post",
        "s2d_pca_budget",
        "avgpool_pca_budget",
        "blurpool_pca_budget",
    ):
        pred = fit_predict_ridge(
            train_reps[rep_name],
            y_train,
            val_reps[rep_name],
            alpha=float(ridge_alpha),
        )
        predictions[rep_name] = pred
        metrics[rep_name] = summarize_prediction(pred, y_val)

    nonlinear_robustness = None
    if run_mlp:
        nonlinear_robustness = {}
        for rep_name in ("pre_pack", "native_post", "s2d_pca_budget"):
            pred_mlp = fit_predict_mlp(
                train_reps[rep_name],
                y_train,
                val_reps[rep_name],
                hidden=64,
                epochs=int(mlp_epochs),
                batch_size=1024,
                seed=_stable_seed(seed, "mlp", rep_name),
                device=probe_device,
            )
            nonlinear_robustness[rep_name] = summarize_prediction(pred_mlp, y_val)

    losses = {name: per_cell_child_mae(pred, y_val) for name, pred in predictions.items()}
    comparisons = {}
    for name, baseline in (
        ("native_minus_pre", "pre_pack"),
        ("native_minus_s2d_pca", "s2d_pca_budget"),
        ("native_minus_avgpool_pca", "avgpool_pca_budget"),
        ("native_minus_blurpool_pca", "blurpool_pca_budget"),
    ):
        comparisons[name] = {
            "all_active": bootstrap_image_mean_difference(
                losses["native_post"], losses[baseline], image_ids,
                n_boot=bootstrap, seed=_stable_seed(seed, name, "all"),
            ),
            "n2plus": bootstrap_image_mean_difference(
                losses["native_post"][multi], losses[baseline][multi], image_ids[multi],
                n_boot=bootstrap, seed=_stable_seed(seed, name, "n2plus"),
            ) if multi.any() else None,
        }

    pre_n2 = metrics["pre_pack"]["n2p_child_mae"]
    native_n2 = metrics["native_post"]["n2p_child_mae"]
    pca_n2 = metrics["s2d_pca_budget"]["n2p_child_mae"]
    rel_native_vs_pre = _safe_ratio(native_n2 - pre_n2, pre_n2)
    rel_native_vs_pca = _safe_ratio(native_n2 - pca_n2, pca_n2)

    native_pre_gap = native_n2 - pre_n2
    closure = {}
    for control in ("s2d_pca_budget", "avgpool_pca_budget", "blurpool_pca_budget"):
        control_n2 = metrics[control]["n2p_child_mae"]
        closure[control] = _safe_ratio(native_n2 - control_n2, native_pre_gap)

    linkage = None
    downstream_pred = val["downstream_parent_pred"]
    if torch.isfinite(downstream_pred).all():
        gt_parent = y_val.sum(dim=1)
        downstream_signed = downstream_pred - gt_parent
        downstream_abs = downstream_signed.abs()
        representation_excess = losses["native_post"] - losses["s2d_pca_budget"]
        if multi.sum() >= 3:
            rho_abs = spearmanr(
                representation_excess[multi].numpy(), downstream_abs[multi].numpy()
            ).statistic
            # Positive means more representation loss is associated with stronger under-counting.
            underestimate = (gt_parent - downstream_pred)
            rho_under = spearmanr(
                representation_excess[multi].numpy(), underestimate[multi].numpy()
            ).statistic
            linkage = {
                "n2plus_cells": int(multi.sum()),
                "spearman_excess_vs_abs_local_count_error": float(rho_abs),
                "spearman_excess_vs_underestimate": float(rho_under),
            }

    cmp_pre = comparisons["native_minus_pre"]["n2plus"]
    cmp_pca = comparisons["native_minus_s2d_pca"]["n2plus"]
    statistical_pre = bool(cmp_pre is not None and cmp_pre["ci95_low"] > 0.0)
    statistical_pca = bool(cmp_pca is not None and cmp_pca["ci95_low"] > 0.0)
    effect = bool(math.isfinite(rel_native_vs_pre) and rel_native_vs_pre >= min_relative_degradation)
    screen_go = statistical_pre and statistical_pca and effect

    linkage_go = None
    if linkage is not None:
        rho_abs = linkage["spearman_excess_vs_abs_local_count_error"]
        rho_under = linkage["spearman_excess_vs_underestimate"]
        linkage_go = bool(
            (math.isfinite(rho_abs) and rho_abs >= min_linkage_rho)
            or (math.isfinite(rho_under) and rho_under >= min_linkage_rho)
        )

    return {
        "projection_controls": projection_meta,
        "metrics": metrics,
        "comparisons": comparisons,
        "relative_n2plus_degradation": {
            "native_vs_pre": rel_native_vs_pre,
            "native_vs_s2d_pca": rel_native_vs_pca,
        },
        "known_control_gap_closure_fraction_n2plus": closure,
        "downstream_linkage": linkage,
        "nonlinear_probe_robustness": nonlinear_robustness,
        "decision": {
            "min_relative_degradation": float(min_relative_degradation),
            "min_linkage_rho": float(min_linkage_rho),
            "native_worse_than_pre_ci95": statistical_pre,
            "native_worse_than_s2d_pca_ci95": statistical_pca,
            "relative_effect_pass": effect,
            "screen_go": screen_go,
            "linkage_go": linkage_go,
            "final_go": bool(screen_go and linkage_go) if linkage_go is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0-v2: transition-wise cardinality sufficiency / compression attribution audit"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--backbone-override",
        default=None,
        help="Diagnostic-only timm pretrained backbone for a cross-architecture control.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--crops-per-image", type=int, default=3)
    p.add_argument("--max-train-cells-per-bin", type=int, default=20000)
    p.add_argument("--max-val-cells-per-bin", type=int, default=10000)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-relative-degradation", type=float, default=0.10)
    p.add_argument("--min-linkage-rho", type=float, default=0.20)
    p.add_argument("--run-mlp", action="store_true", help="Secondary nonlinear-accessibility robustness probe")
    p.add_argument("--mlp-epochs", type=int, default=40)
    p.add_argument("--probe-device", default="cpu", help="Device for optional MLP probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-images", type=int, default=None, help="Smoke/debug only; do not use for final claims")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(args.device)
    source = load_feature_source(
        cfg,
        device=device,
        checkpoint=args.checkpoint,
        backbone_override=args.backbone_override,
    )
    dataset = build_train_dataset(
        cfg,
        image_mean=source.normalization_mean,
        image_std=source.normalization_std,
    )
    train_indices, val_indices = split_image_indices(
        dataset.image_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if args.max_images is not None:
        # Debug-only cap is applied independently so both partitions remain non-empty.
        limit = max(1, int(args.max_images))
        train_indices = train_indices[:limit]
        val_indices = val_indices[: max(1, min(limit, len(val_indices)))]

    print(
        f"E0-v2 source={source.provenance['kind']} device={device} "
        f"images train/val={len(train_indices)}/{len(val_indices)} "
        f"reductions={source.reductions}",
        flush=True,
    )

    train_collected = collect_split(
        dataset,
        train_indices,
        source,
        device,
        crops_per_image=args.crops_per_image,
        max_cells_per_bin=args.max_train_cells_per_bin,
        seed=_stable_seed(args.seed, "train"),
    )
    val_collected = collect_split(
        dataset,
        val_indices,
        source,
        device,
        crops_per_image=args.crops_per_image,
        max_cells_per_bin=args.max_val_cells_per_bin,
        seed=_stable_seed(args.seed, "val"),
    )

    results = {}
    for transition in sorted(train_collected):
        key = f"C{transition[0]}_to_C{transition[1]}"
        results[key] = {
            "train_cell_counts": train_collected[transition]["counts"],
            "val_cell_counts": val_collected[transition]["counts"],
            **evaluate_transition(
                train_collected[transition],
                val_collected[transition],
                ridge_alpha=args.ridge_alpha,
                bootstrap=args.bootstrap,
                seed=_stable_seed(args.seed, key),
                min_relative_degradation=args.min_relative_degradation,
                min_linkage_rho=args.min_linkage_rho,
                run_mlp=args.run_mlp,
                mlp_epochs=args.mlp_epochs,
                probe_device=args.probe_device,
            ),
        }

    payload = {
        "protocol": {
            "name": "E0-v2 Compression Attribution Audit",
            "repo_git_sha": _git_sha(),
            "config": str(args.config),
            "checkpoint": args.checkpoint,
            "backbone_override": args.backbone_override,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "crops_per_image": args.crops_per_image,
            "max_train_cells_per_bin": args.max_train_cells_per_bin,
            "max_val_cells_per_bin": args.max_val_cells_per_bin,
            "ridge_alpha": args.ridge_alpha,
            "bootstrap": args.bootstrap,
            "run_mlp": args.run_mlp,
            "mlp_epochs": args.mlp_epochs,
            "probe_device": args.probe_device,
            "train_image_count": len(train_indices),
            "val_image_count": len(val_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "source": source.provenance,
            "normalization_mean": list(source.normalization_mean),
            "normalization_std": list(source.normalization_std),
            "important_note": (
                "PCA is a budget-matched linear control, not a theoretical information upper bound. "
                "Blur/average pooling controls are diagnostics, not proposed contributions."
            ),
        },
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Wrote {output}", flush=True)
    for key, value in results.items():
        d = value["decision"]
        print(
            f"{key}: screen_go={d['screen_go']} linkage_go={d['linkage_go']} final_go={d['final_go']} "
            f"native/pre n2+={value['relative_n2plus_degradation']['native_vs_pre']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

```

---

# 4. Full code — `tools/summarize_cardinality_sufficiency_v2.py`

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Summarize E0-v2 JSON runs into one CSV")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        protocol = payload["protocol"]
        source = protocol.get("source", {})
        source_name = source.get("kind", "unknown")
        source_model = source.get("model") or source.get("pretrained_spec", {}).get("architecture") or "current"
        for transition, result in payload["results"].items():
            decision = result["decision"]
            closure = result["known_control_gap_closure_fraction_n2plus"]
            linkage = result.get("downstream_linkage") or {}
            for rep_name, metrics in result["metrics"].items():
                rows.append(
                    {
                        "file": str(input_path),
                        "source_kind": source_name,
                        "source_model": source_model,
                        "transition": transition,
                        "representation": rep_name,
                        "cells": metrics.get("cells"),
                        "child_mae": metrics.get("child_mae"),
                        "parent_mae": metrics.get("parent_mae"),
                        "composition_l1": metrics.get("composition_l1"),
                        "n2p_cells": metrics.get("n2p_cells"),
                        "n2p_child_mae": metrics.get("n2p_child_mae"),
                        "n2p_parent_mae": metrics.get("n2p_parent_mae"),
                        "n2p_composition_l1": metrics.get("n2p_composition_l1"),
                        "relative_native_vs_pre_n2p": result["relative_n2plus_degradation"].get("native_vs_pre"),
                        "relative_native_vs_s2d_pca_n2p": result["relative_n2plus_degradation"].get("native_vs_s2d_pca"),
                        "s2d_pca_gap_closure": closure.get("s2d_pca_budget"),
                        "avgpool_pca_gap_closure": closure.get("avgpool_pca_budget"),
                        "blurpool_pca_gap_closure": closure.get("blurpool_pca_budget"),
                        "rho_excess_vs_abs_local_error": linkage.get("spearman_excess_vs_abs_local_count_error"),
                        "rho_excess_vs_underestimate": linkage.get("spearman_excess_vs_underestimate"),
                        "screen_go": decision.get("screen_go"),
                        "linkage_go": decision.get("linkage_go"),
                        "final_go": decision.get("final_go"),
                    }
                )
    if not rows:
        raise RuntimeError("No rows found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

```

---

# 5. Full code — `tests/test_cardinality_sufficiency_v2.py`

```python
import torch

from cardinality_sufficiency_v2 import (
    PCAProjector,
    avgpool2x,
    blurpool2x,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_ridge,
    pack_2x2_features,
    pack_child_counts,
    summarize_prediction,
)


def test_pack_geometry_and_order():
    x = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    packed = pack_2x2_features(x)
    assert packed.shape == (1, 2, 2, 4)
    assert torch.equal(packed[0, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))

    y = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    child = pack_child_counts(y)
    assert torch.equal(child[0, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_lossless_pack_contains_all_values():
    x = torch.randn(2, 3, 8, 10)
    packed = pack_2x2_features(x)
    assert packed.numel() == x.numel()
    assert torch.allclose(
        torch.sort(packed.reshape(-1)).values,
        torch.sort(x.reshape(-1)).values,
    )


def test_pool_controls_have_expected_shape():
    x = torch.randn(2, 5, 8, 10)
    assert avgpool2x(x).shape == (2, 5, 4, 5)
    assert blurpool2x(x).shape == (2, 5, 4, 5)


def test_representation_grid_alignment():
    pre = torch.randn(1, 6, 16, 16)
    post = torch.randn(1, 10, 8, 8)
    grids = build_representation_grid(pre, post)
    assert grids["pre_pack"].shape == (1, 8, 8, 24)
    assert grids["native_post"].shape == (1, 8, 8, 10)
    assert grids["avgpool"].shape == (1, 8, 8, 6)
    assert grids["blurpool"].shape == (1, 8, 8, 6)


def test_pca_budget_dimension():
    x = torch.randn(100, 12)
    pca = PCAProjector.fit(x, output_dim=7)
    assert pca.transform(x).shape == (100, 7)
    pca_expand = PCAProjector.fit(x, output_dim=15)
    z = pca_expand.transform(x)
    assert z.shape == (100, 15)
    assert torch.count_nonzero(z[:, 12:]) == 0


def test_ridge_recovers_linear_child_counts():
    torch.manual_seed(0)
    x_train = torch.randn(400, 20)
    w = torch.randn(20, 4)
    y_train = x_train @ w + 0.01 * torch.randn(400, 4)
    x_val = torch.randn(100, 20)
    y_val = x_val @ w
    pred = fit_predict_ridge(x_train, y_train, x_val, alpha=1e-3)
    assert (pred - y_val).abs().mean() < 0.02


def test_summary_has_dense_bins():
    y = torch.tensor(
        [[1,0,0,0], [1,1,0,0], [2,1,1,0], [2,2,1,1]], dtype=torch.float32
    )
    pred = y.clone()
    m = summarize_prediction(pred, y)
    assert m["child_mae"] == 0.0
    assert m["n2p_cells"] == 3
    assert m["n5p_cells"] == 1


def test_bootstrap_diff_sign():
    a = torch.tensor([2.0, 2.0, 4.0, 4.0])
    b = torch.tensor([1.0, 1.0, 2.0, 2.0])
    ids = torch.tensor([0, 0, 1, 1])
    out = bootstrap_image_mean_difference(a, b, ids, n_boot=200, seed=1)
    assert out["mean_diff"] > 0


def test_tiny_mlp_probe_runs():
    from cardinality_sufficiency_v2 import fit_predict_mlp
    torch.manual_seed(3)
    x = torch.randn(120, 6)
    y = torch.relu(x[:, :4])
    pred = fit_predict_mlp(
        x[:100], y[:100], x[100:], hidden=16, epochs=3, batch_size=32, seed=3
    )
    assert pred.shape == (20, 4)
    assert torch.isfinite(pred).all()

```

The standalone diagnostic unit suite was smoke-tested before this document was generated:

```text
9 passed
```

This validates tensor geometry, exact 2×2 packing order, lossless packing value preservation, BlurPool/AvgPool output geometry, PCA output budget, ridge recovery, metric binning, image-level bootstrap code, and the optional nonlinear probe execution.

It does **not** substitute for running the repository's complete test suite after integration.

---

# 6. Optional `hpc/diagnostics/__init__.py` edit

Not required. If you want explicit package exports, append:

```python
from .cardinality_sufficiency_v2 import (
    PCAProjector,
    TinyMLPProbe,
    avgpool2x,
    blurpool2x,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_mlp,
    fit_predict_ridge,
    pack_2x2_features,
    pack_child_counts,
    summarize_prediction,
)
```

Do not remove the existing D-R / D-K / D-L / D-M exports.

---

# 7. Install and verify

From repository root:

```bash
git checkout feat/ntpc-neural-tree-polya
git rev-parse HEAD
```

The implementation in this document was written against:

```text
753ceb180ee4f1e20cd00924b44013e7479e7398
```

After adding the four files:

```bash
pytest -q tests/test_cardinality_sufficiency_v2.py
```

Then run the full existing suite:

```bash
pytest -q
```

Do not continue to scientific experiments if the existing suite regresses.

---

# 8. Dataset protocol — important

E0-v2 **does not use the official test split to select the hypothesis**.

For ShanghaiTech A, QNRF, or NWPU:

- load only the official **training** set;
- split training images deterministically into diagnostic-train and diagnostic-val;
- default: 80/20 image-level split;
- multiple crops from one source image never cross partitions;
- crop/scale/flip randomness is deterministic per `(image_id, crop_id, seed)`.

This avoids cell-level and crop-level leakage.

The primary first screen should be **ShanghaiTech Part A**, because that is the current carrier's most established protocol. QNRF/NWPU are confirmation datasets only after the mechanism survives the first gate.

---

# 9. Run 1 — current pretrained carrier, no crowd-task checkpoint

Use the exact current Cell-A-style config:

```bash
python tools/run_cardinality_sufficiency_v2.py   --config configs/factorial_a_crop256_c16.yaml   --output runs/e0_v2/pretrained_mobilev4s05.json   --seed 42   --val-fraction 0.20   --crops-per-image 3   --max-train-cells-per-bin 20000   --max-val-cells-per-bin 10000   --ridge-alpha 1.0   --bootstrap 2000
```

This answers:

> Is the cardinality accessibility loss already present in the ImageNet-pretrained compact representation?

Do **not** use this run alone to claim a crowd-counting bottleneck, because it has no downstream crowd-task error linkage.

---

# 10. Run 2 — task-trained current carrier

Use the scientifically selected checkpoint corresponding to the same config.

Example:

```bash
python tools/run_cardinality_sufficiency_v2.py   --config configs/factorial_a_crop256_c16.yaml   --checkpoint runs/factorial_a_crop256_c16/best.pt   --output runs/e0_v2/tasktrained_mobilev4s05.json   --seed 42   --val-fraction 0.20   --crops-per-image 3   --max-train-cells-per-bin 20000   --max-val-cells-per-bin 10000   --ridge-alpha 1.0   --bootstrap 2000
```

When a checkpoint is supplied, the runner also evaluates local downstream count error from the model's stride-4 mass map.

For transition `s → 2s`, the mass map is exactly sum-pooled to the parent stride `2s`, so downstream local error is aligned with the same region used by the cardinality probe.

The runner then reports:

```text
Spearman(
    native-cardinality-excess-loss,
    absolute-local-count-error
)

Spearman(
    native-cardinality-excess-loss,
    local-undercount
)
```

This is necessary because a representation diagnostic that does not predict actual counting failure is insufficient for the architectural claim.

---

# 11. Run 3 — optional nonlinear probe robustness check

The primary probe is ridge regression because it is deterministic, cheap, and interpretable.

If the ridge audit passes, run the fixed tiny MLP robustness probe:

```bash
python tools/run_cardinality_sufficiency_v2.py   --config configs/factorial_a_crop256_c16.yaml   --checkpoint runs/factorial_a_crop256_c16/best.pt   --output runs/e0_v2/tasktrained_mobilev4s05_mlp.json   --run-mlp   --mlp-epochs 40   --probe-device cuda   --seed 42
```

The MLP:

- uses one hidden layer of width 64;
- uses the same fixed training protocol for all representations;
- has no validation-driven early stopping;
- is a robustness analysis, not the primary metric.

Interpretation:

- if ridge says `native_post` is much worse but the MLP completely erases the gap, the evidence may be **nonlinearly present rather than destroyed**;
- do not call this irreversible information loss.

Use the safer wording:

> **reduced accessibility of local cardinality evidence under the tested probe family**

until stronger causal evidence exists.

---

# 12. Run 4 — stronger pretrained backbone control

E0-v2 can use a diagnostic-only generic timm feature extractor without changing the production model.

First list the installed/pretrained candidates:

```bash
python - <<'PY'
import timm
for name in timm.list_models(pretrained=True):
    if "mobilenetv4" in name.lower():
        print(name)
PY
```

Select a meaningfully stronger model that exposes reductions 4, 8, and 16.

Then:

```bash
python tools/run_cardinality_sufficiency_v2.py   --config configs/factorial_a_crop256_c16.yaml   --backbone-override <EXACT_TIMM_MODEL_NAME>   --output runs/e0_v2/strong_pretrained_control.json   --seed 42   --val-fraction 0.20   --crops-per-image 3
```

The generic extractor automatically switches image normalization to the selected timm pretrained weight metadata while keeping the **same deterministic image/crop split geometry**.

Important limitation:

> A generic ImageNet-pretrained strong backbone is only a **representation control**.  
> For the final paper, if E0 survives, a stronger **crowd-task-trained** control is still required.

---

# 13. Aggregate the runs

```bash
python tools/summarize_cardinality_sufficiency_v2.py   runs/e0_v2/pretrained_mobilev4s05.json   runs/e0_v2/tasktrained_mobilev4s05.json   runs/e0_v2/strong_pretrained_control.json   --output runs/e0_v2/e0_v2_summary.csv
```

The CSV contains one row per:

```text
source × transition × representation
```

with:

- child MAE;
- parent MAE;
- normalized 4-child composition L1;
- N≥2 metrics;
- relative native-vs-pre degradation;
- relative native-vs-PCA-budget degradation;
- gap closure by S2D+PCA / AvgPool+PCA / BlurPool+PCA;
- downstream Spearman linkage;
- screen/final GO flags.

---

# 14. What each representation means

## `pre_pack`

```text
2×2 C_s neighborhood
→ exact TL/TR/BL/BR packing
→ 4*C_s vector
```

No feature value is discarded.

It is a pre-transition accessibility reference, **not a budget-matched architecture**.

---

## `native_post`

Actual current backbone output at `C_2s`.

This contains whatever spatial filtering, striding, nonlinear transformation, and channel compression the timm MobileNetV4 stage performs.

---

## `s2d_pca_budget`

```text
C_s
→ lossless 2×2 packing
→ PCA fitted on diagnostic-train only
→ exactly C_post dimensions
```

This tests whether much of the native degradation is simply forced by the reduced representation dimension.

Do not describe PCA as “optimal for cardinality”; it is not target supervised.

---

## `avgpool_pca_budget`

```text
C_s
→ average-pool stride 2
→ train-only PCA
→ C_post dimensions
```

A plain spatial decimation control.

---

## `blurpool_pca_budget`

```text
C_s
→ fixed [1,2,1]×[1,2,1] low-pass
→ stride 2
→ train-only PCA
→ C_post dimensions
```

A frozen anti-aliasing diagnostic control.

It is **not** an implementation of a full BlurPool-trained backbone and must not be described as such in a paper.

---

# 15. Metrics

The exact target for each parent region is

\[
Y_R=[n_{TL},n_{TR},n_{BL},n_{BR}].
\]

Primary metric:

\[
E_{child}
=
\frac14
\sum_{j=1}^4
|\hat n_j-n_j|.
\]

Also report parent-total probe error:

\[
E_{parent}
=
\left|
\sum_j\hat n_j
-
\sum_j n_j
\right|.
\]

And composition error for active parents:

\[
E_{comp}
=
\left\|
\frac{\hat Y_+}{\sum_j\hat Y_{+,j}}
-
\frac{Y}{\sum_jY_j}
\right\|_1,
\]

where negative predictions are clipped to zero only for the normalized composition metric.

Stratify by exact parent multiplicity:

```text
N = 1
N = 2
N = 3–4
N >= 5
N >= 2 combined
```

The **N≥2** result is the primary crowd-specific gate.

Do not infer tiny-head behavior from ShanghaiTech A because the current point-only target does not provide reliable per-head physical size labels.

---

# 16. Statistical unit

Do not bootstrap individual cells.

Cells from the same image/crop are correlated.

The code therefore:

1. computes probe error per cell;
2. groups paired representation differences by source image;
3. computes the image-level mean difference;
4. bootstraps images.

The reported CI is therefore an **image-level paired bootstrap CI**.

---

# 17. Pre-registered GO / NO-GO screen

The code defaults to:

```text
minimum relative N>=2 degradation = 10%
minimum downstream Spearman rho   = 0.20
```

These are screening thresholds, not universal scientific constants. Freeze them before looking at the full result; do not move them after seeing the outcome.

For a transition to pass `screen_go`, all three must hold:

1. image-bootstrap 95% CI for

   ```text
   native N>=2 child-MAE - pre_pack N>=2 child-MAE
   ```

   is entirely above zero;

2. image-bootstrap 95% CI for

   ```text
   native N>=2 child-MAE - s2d_pca_budget N>=2 child-MAE
   ```

   is entirely above zero;

3. native N≥2 child-MAE is at least 10% worse than `pre_pack`.

With a task-trained checkpoint, `final_go` additionally requires positive downstream linkage:

```text
rho >= 0.20
```

for at least one of:

- representation excess vs absolute local count error;
- representation excess vs local undercount.

---

# 18. Interpretation matrix

## Case A — native ≈ pre

```text
native_post ≈ pre_pack
```

**NO-GO.**

There is no meaningful transition-wise cardinality accessibility loss.

Do not design a new downsampler.

---

## Case B — native worse, but native ≈ S2D+PCA

```text
pre_pack good
S2D+PCA worse
native ≈ S2D+PCA
```

Likely interpretation:

> the fixed output-dimensionality/channel budget explains most of the degradation.

The research question should move toward **count-relevant channel compression**, not spatial anti-aliasing.

Do not claim a spatial-downsampling mechanism.

---

## Case C — native worse than S2D+PCA

```text
pre_pack best
S2D+PCA intermediate
native clearly worse
```

This is the interesting outcome.

It says that the native transition destroys/accessibly suppresses more cardinality evidence than a simple budget-matched linear compression control.

Only this case justifies studying the internal transition mechanism.

---

## Case D — BlurPool control closes most of the gap

Define gap closure approximately as

\[
\text{closure}
=
\frac{E_{native}-E_{control}}
{E_{native}-E_{pre}}.
\]

If `blurpool_pca_budget` closes roughly 80% or more of the native-pre gap:

> anti-aliasing already explains most of the observed failure.

Treat that as a **novelty danger / likely kill**, not as evidence to rename BlurPool.

A proper full-model BlurPool/APS/TIPS comparison would then be required before any new architecture.

---

## Case E — diagnostic gap exists but no downstream linkage

```text
representation loss != actual counting failure
```

**NO-GO for the causal architecture paper.**

The representation phenomenon may be real but is not yet shown to matter for counting.

---

## Case F — compact model fails, stronger control does not

This is the strongest desired pattern:

```text
compact:
    pre good
    native bad
    excess native-vs-budget > 0

strong control:
    much smaller excess degradation
```

Combined with downstream linkage, this supports a genuinely lightweight-specific compression bottleneck.

---

# 19. Why global count conservation is not enough

Do not write:

> “Global count supervision is insufficient for counting.”

That is false wording.

Use:

> **Global count conservation constrains total mass but does not uniquely constrain local cardinality composition.**

For example:

\[
[1,1,1,1]
\quad\text{and}\quad
[0,0,0,4]
\]

both have total count 4.

Therefore

\[
\sum_jY_j=N
\]

does not identify the local 4-child composition \(Y\).

This is a mathematical non-identifiability statement; it is **not** a claim that a density counter must localize every individual head.

---

# 20. Do not add `L_card` yet

Local/window/grid counting losses already have substantial prior art.

E0-v2 uses exact local counts as **diagnostic targets**, not as a proposed training loss.

Do not modify:

```text
hpc/losses/ntpc.py
hpc/losses/factory.py
train_ntpc.py
```

to add `L_card` until there is a separate novelty argument and a causal necessity demonstrated by E0/E1.

---

# 21. Do not add a bypass yet

Do not add a C4/C8 bypass, extra FPN route, or residual connection as part of E0.

If E0 later identifies `C8→C16` as the only damaging transition, a bypass becomes one **known control** in E1, not an assumed contribution.

A generic bypass winning immediately is a novelty warning.

---

# 22. Do not call S2D/PCA “SPD-Conv”

`pre_pack` and `s2d_pca_budget` are **offline representation controls**.

They do not implement a trained SPD-Conv block.

If E0-v2 is GO, the next causal experiment must include an actual matched, retrained known downsampling baseline such as:

```text
native stride transition
vs
anti-aliased transition
vs
SPD-style transition
vs
other relevant known learnable/polyphase transition
```

with matched:

- backbone stage position;
- output channels;
- training schedule;
- pretrained initialization policy where possible;
- crop protocol;
- parameter/FLOP reporting;
- random seeds.

Only after known controls fail to close the mechanism-specific gap should a new operator be designed.

---

# 23. E1 only if E0-v2 passes

The next stage is **not**:

```text
Baseline
+ new downsampler
+ bypass
+ local loss
```

Instead:

```text
E0-v2 identifies transition + failure type
        |
        v
E1 causal controls
  - native
  - anti-alias
  - S2D/SPD-style
  - narrow bypass
        |
        v
Does one known control explain the effect?
   yes -> kill/reframe
   no  -> inspect residual mechanism
        |
        v
Only then design a new operator
```

The architecture must come from the residual causal failure.

---

# 24. Recommended run order

Use this order exactly:

```text
A. Unit tests
B. Pretrained current compact carrier
C. Task-trained current compact carrier
D. Optional MLP robustness
E. Strong pretrained representation control
F. Repeat seed split 1–2 times only if the primary result is near the gate
G. If GO: QNRF/NWPU confirmation
H. If still GO: E1 full-model known downsampler controls
I. Only then new architecture
```

Do not spend 1500 epochs on a new architecture before step H establishes that the residual problem is not already solved by known operators.

---

# 25. Minimal expected output structure

Example structure, not expected numerical values:

```json
{
  "protocol": {
    "name": "E0-v2 Compression Attribution Audit",
    "source": {
      "kind": "task_checkpoint"
    }
  },
  "results": {
    "C4_to_C8": {
      "metrics": {
        "pre_pack": {},
        "native_post": {},
        "s2d_pca_budget": {},
        "avgpool_pca_budget": {},
        "blurpool_pca_budget": {}
      },
      "comparisons": {
        "native_minus_pre": {
          "n2plus": {
            "mean_diff": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0
          }
        }
      },
      "downstream_linkage": {},
      "decision": {
        "screen_go": false,
        "linkage_go": null,
        "final_go": null
      }
    }
  }
}
```

Never substitute example values for real results.

---

# 26. Important repository-specific notes

1. Current production backbone already returns a configured tuple of `C4/C8/C16[/C32]`; no hook surgery is necessary for E0.
2. Current point pipeline already constructs exact integer `Y4/Y8/Y16/Y32/Y64`, so the audit does not need Gaussian density targets.
3. The current mass head outputs stride-4 positive mass, allowing exact non-overlapping block sums for downstream local-count linkage.
4. The model factory already has checkpoint compatibility checks; E0-v2 uses them before loading a task checkpoint.
5. Pretrained normalization must remain matched to the exact timm weight configuration.
6. The generic timm control is deliberately diagnostic-only and does not modify the deployed `MobileNetV4Backbone`.

---

# 27. Scientific claims allowed after E0-v2

Before any result:

> We evaluate whether local cardinality evidence becomes less accessible across spatial-compression transitions in an ultra-light crowd counter.

If the native-vs-pre gap is significant:

> Local cardinality is less linearly decodable after the tested transition.

If native is also worse than S2D+PCA at the same output dimension:

> The observed degradation is not explained solely by a simple budget-matched linear dimensionality reduction control.

If the nonlinear probe agrees:

> The effect is robust to both linear and small nonlinear accessibility probes.

If downstream linkage also holds:

> Regions exhibiting larger transition-specific cardinality accessibility loss are associated with larger downstream local counting errors/undercounting.

Do **not** write “information is irreversibly destroyed” without substantially stronger evidence.

---

# 28. What would kill the entire direction

Immediately stop the downsampling architecture direction if any of these occurs consistently:

- `native_post` is not worse than `pre_pack`;
- `native_post` and `s2d_pca_budget` are effectively equivalent;
- the effect exists only with a linear probe and vanishes with the fixed MLP probe;
- the effect does not predict downstream counting error;
- the compact model and stronger control show the same effect;
- BlurPool/known downsampling controls explain nearly all of the residual failure;
- results are driven only by one image split/seed;
- the effect is present only on N=1 cells and disappears for N≥2.

In those cases, do not rescue the hypothesis by adding attention, an FPN route, a local-count loss, or a new name.

---

# 29. Deliverables to keep after the run

Keep:

```text
runs/e0_v2/*.json
runs/e0_v2/e0_v2_summary.csv
exact YAML config(s)
checkpoint SHA / epoch
git SHA
timm model/weight identifiers
train/val image index lists embedded in JSON
```

These are sufficient to reconstruct the diagnostic split and compare future controls.

---

# 30. Final decision rule

The direction earns an E1 architecture experiment only if the evidence chain is:

\[
\boxed{
\text{pre accessible}
\rightarrow
\text{native degradation}
\rightarrow
\text{degradation exceeds budget control}
\rightarrow
\text{N≥2 specific}
\rightarrow
\text{downstream linkage}
\rightarrow
\text{strong-control specificity}
}
\]

If any central link fails, reformulate or kill the hypothesis before expensive training.
