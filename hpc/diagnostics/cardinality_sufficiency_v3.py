from __future__ import annotations

"""E0-v3 Cardinality Sufficiency Diagnostics — Methodologically corrected version.

Changes from v2:
* Unified image-weighted estimand: effect size AND CI both use image-level averaging.
  No cell-weighted pooling is used for any primary metric.
* Parent cardinality MAE and child composition L1 are tracked as independent scalars
  with independent bootstrap CIs — never conflated.
* BlurPool / AvgPool helper functions are kept for informative decimation controls
  (reported separately, excluded from GO/NO-GO).
* ``summarize_prediction`` is retained for bin-stratified diagnostics only,
  not used in the primary decision path.
* Paired multi-seed MLP evaluation via ``paired_seed_mlp_eval``.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


BIN_NAMES = ("n1", "n2", "n3_4", "n5p")


# ---------------------------------------------------------------------------
# Feature geometry helpers (unchanged from v2)
# ---------------------------------------------------------------------------

def pack_2x2_features(x: torch.Tensor) -> torch.Tensor:
    """Losslessly pack each non-overlapping 2x2 feature neighbourhood.

    Input:  [B, C, H, W]
    Output: [B, H/2, W/2, 4C] in TL, TR, BL, BR order.

    SpaceToDepth(2) rearrangement — information-preserving up to tensor reordering.
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
    """Convert [B,C,H,W] → [B,H,W,C]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    return x.permute(0, 2, 3, 1).contiguous()


def pack_child_counts(y: torch.Tensor) -> torch.Tensor:
    """Pack four-child counts for stride-s → stride-2s transition.

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


# ---------------------------------------------------------------------------
# Informative decimation controls — NOT used in GO/NO-GO (see runner)
# ---------------------------------------------------------------------------

def blurpool2x(x: torch.Tensor) -> torch.Tensor:
    """Fixed anti-aliased 2x decimation — informative control only.

    CAUTION: when input channels < output budget, zero-padding is required.
    The runner must report ``effective_rank`` and must NOT include this in
    the primary GO/NO-GO decision.
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
    """Plain 2x average-pooling — informative control only.

    CAUTION: same budget-mismatch caveat as blurpool2x.
    """
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


# ---------------------------------------------------------------------------
# Standardizer and PCA (unchanged from v2)
# ---------------------------------------------------------------------------

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
    effective_rank: int  # NEW: explicitly tracked; runner must log this

    @classmethod
    def fit(cls, x: torch.Tensor, output_dim: int) -> "PCAProjector":
        """Fit PCA on [N,D].

        If output_dim > D, only D components are retained — the runner must
        report effective_rank so the caller can flag dimension mismatches.
        Zero-padding is applied in transform() only when output_dim > D.
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
        effective_rank = min(int(output_dim), int(x.shape[1]))
        order = torch.argsort(eigvals, descending=True)[:effective_rank]
        basis = eigvecs[:, order].contiguous()
        return cls(mean=mean, basis=basis, output_dim=int(output_dim), effective_rank=effective_rank)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.double() - self.mean) @ self.basis
        if z.shape[1] < self.output_dim:
            pad = z.new_zeros((z.shape[0], self.output_dim - z.shape[1]))
            z = torch.cat((z, pad), dim=1)
        return z.float()


# ---------------------------------------------------------------------------
# Ridge probe (unchanged from v2)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-cell error primitives (unchanged, used for bin-stratified diagnostics)
# ---------------------------------------------------------------------------

def per_cell_parent_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    return (pred.float().sum(dim=1) - target.float().sum(dim=1)).abs()


def per_cell_composition_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L1 error between normalised four-child compositions, active parents only."""
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    p = pred.float().clamp_min(0.0)
    t = target.float()
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(eps)
    return (p - t).abs().sum(dim=1)


def summarize_prediction(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Bin-stratified summary for diagnostic purposes.

    Note: uses cell-level averaging within each bin, NOT the primary image-weighted
    estimand. Do not use this for GO/NO-GO decisions in E0-v3.
    """
    parent = per_cell_parent_mae(pred, target)
    composition = per_cell_composition_l1(pred, target)
    result: Dict[str, float] = {
        "cells": int(target.shape[0]),
        "parent_mae_cell_avg": float(parent.mean()) if parent.numel() else float("nan"),
        "composition_l1_cell_avg": float(composition.mean()) if composition.numel() else float("nan"),
        "negative_prediction_fraction": float((pred < 0).float().mean()) if pred.numel() else float("nan"),
    }
    n = target.sum(dim=1)
    labels = parent_count_bin(n)
    for idx, name in enumerate(BIN_NAMES):
        mask = labels == idx
        result[f"{name}_cells"] = int(mask.sum())
        result[f"{name}_parent_mae"] = float(parent[mask].mean()) if mask.any() else float("nan")
        result[f"{name}_composition_l1"] = float(composition[mask].mean()) if mask.any() else float("nan")
    multi = n >= 2
    result["n2p_cells"] = int(multi.sum())
    result["n2p_parent_mae"] = float(parent[multi].mean()) if multi.any() else float("nan")
    result["n2p_composition_l1"] = float(composition[multi].mean()) if multi.any() else float("nan")
    return result


# ---------------------------------------------------------------------------
# Primary image-weighted estimand (NEW in v3)
# ---------------------------------------------------------------------------

def _per_image_mean(
    values: torch.Tensor,
    image_ids: torch.Tensor,
) -> torch.Tensor:
    """Return per-image mean of a 1-D cell-level tensor.

    Returns a 1-D tensor of shape [n_images] preserving image ordering from
    ``torch.unique(image_ids)``.
    """
    if values.ndim != 1 or image_ids.ndim != 1:
        raise ValueError("values and image_ids must be 1-D")
    if values.numel() != image_ids.numel():
        raise ValueError("values and image_ids must have equal length")
    unique = torch.unique(image_ids.cpu(), sorted=True)
    per_image = torch.stack(
        [values.cpu()[image_ids.cpu() == img_id].mean() for img_id in unique]
    )
    return per_image  # [n_images]


def image_weighted_mean(
    values: torch.Tensor,
    image_ids: torch.Tensor,
) -> float:
    """Image-averaged mean: mean over images of per-image mean.

    This is the primary estimand for E0-v3. Both effect size AND bootstrap CI
    must call this function so they use the same quantity.
    """
    return float(_per_image_mean(values, image_ids).mean())


def image_weighted_bootstrap_diff(
    a: torch.Tensor,
    b: torch.Tensor,
    image_ids: torch.Tensor,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Image-level bootstrap CI for mean_a − mean_b, same estimand as image_weighted_mean.

    Args:
        a, b: 1-D cell-level error tensors (same estimand quantity, e.g., parent_mae).
        image_ids: 1-D integer tensor, cell → image mapping.
        n_boot: number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns dict with keys: mean_diff, ci95_low, ci95_high, images.
    """
    if a.ndim != 1 or b.ndim != 1 or image_ids.ndim != 1:
        raise ValueError("a, b, image_ids must be 1-D")
    if not (a.numel() == b.numel() == image_ids.numel()):
        raise ValueError("a, b, image_ids must have equal length")
    unique = torch.unique(image_ids.cpu(), sorted=True)
    n_images = int(unique.numel())
    if n_images < 2:
        diff_mean = float((a - b).mean())
        return {
            "mean_diff": diff_mean,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "images": n_images,
        }
    # Compute per-image mean difference (same estimand as image_weighted_mean)
    per_image_diff = torch.tensor(
        [
            float(
                a.cpu()[image_ids.cpu() == img_id].mean()
                - b.cpu()[image_ids.cpu() == img_id].mean()
            )
            for img_id in unique.tolist()
        ],
        dtype=torch.float64,
    )
    g = torch.Generator().manual_seed(int(seed))
    boot = torch.empty(int(n_boot), dtype=torch.float64)
    m = per_image_diff.numel()
    for i in range(int(n_boot)):
        idx = torch.randint(0, m, (m,), generator=g)
        boot[i] = per_image_diff[idx].mean()
    q = torch.quantile(boot, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "mean_diff": float(per_image_diff.mean()),
        "ci95_low": float(q[0]),
        "ci95_high": float(q[1]),
        "images": n_images,
    }


def compute_image_weighted_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute separated, image-weighted primary metrics.

    Returns:
        parent_mae_iw:      image-weighted mean of per-cell parent count MAE.
        composition_l1_iw:  image-weighted mean of per-cell composition L1.
        n_images:           number of images contributing.
        n_cells:            total active cells.

    These are the two independent primary hypotheses for E0-v3 GO/NO-GO.
    They MUST NOT be aggregated into a single metric.
    """
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("Expected pred/target [N,4]")
    if image_ids.ndim != 1 or image_ids.numel() != pred.shape[0]:
        raise ValueError("image_ids must be 1-D with same length as pred")
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
        image_ids = image_ids[mask]
    if pred.shape[0] == 0:
        return {
            "parent_mae_iw": float("nan"),
            "composition_l1_iw": float("nan"),
            "n_images": 0,
            "n_cells": 0,
        }
    parent_errors = per_cell_parent_mae(pred, target)
    comp_errors = per_cell_composition_l1(pred, target)
    n_images = int(torch.unique(image_ids).numel())
    return {
        "parent_mae_iw": image_weighted_mean(parent_errors, image_ids),
        "composition_l1_iw": image_weighted_mean(comp_errors, image_ids),
        "n_images": n_images,
        "n_cells": int(pred.shape[0]),
    }


# ---------------------------------------------------------------------------
# Uniform image cell collector (replaces TransitionReservoir)
# ---------------------------------------------------------------------------

class ImageCellCollector:
    """Uniform-per-image active-cell collector — fixes v2 reservoir saturation bias.

    Each image contributes at most ``max_cells_per_image`` cells.
    Images are never excluded due to early-arrival saturation; the budget is
    pre-allocated uniformly across all images regardless of order.
    """

    def __init__(self, max_cells_per_image: int, seed: int):
        if max_cells_per_image <= 0:
            raise ValueError("max_cells_per_image must be positive")
        self.max_cells_per_image = int(max_cells_per_image)
        self.seed = int(seed)
        # Per-image storage: image_id → dict of lists
        self._store: dict[int, dict[str, list]] = {}

    def add(
        self,
        reps: Dict[str, torch.Tensor],
        y_children: torch.Tensor,
        image_id: int,
        downstream_parent_pred: torch.Tensor | None = None,
    ) -> None:
        """Add active cells from a single crop of a single image.

        Args:
            reps:   Dict of representation name → [1,H,W,D] tensor.
            y_children: [1,H,W,4] child count tensor.
            image_id: integer image identifier.
            downstream_parent_pred: optional [1,H,W] task-model parent count predictions.
        """
        if y_children.ndim != 4 or y_children.shape[0] != 1 or y_children.shape[-1] != 4:
            raise ValueError("Expected y_children [1,H,W,4]")
        y = y_children.reshape(-1, 4).float().cpu()
        n = y.sum(dim=1)
        # Only retain active parent cells (N >= 1)
        active = n >= 1
        if not active.any():
            return

        flat_reps = {}
        for name, tensor in reps.items():
            if tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError(f"Rep {name} must be [1,H,W,D], got {tuple(tensor.shape)}")
            flat_reps[name] = tensor.reshape(-1, tensor.shape[-1]).float().cpu()[active]

        y_active = y[active]
        n_active = n[active]

        if downstream_parent_pred is not None:
            ds = downstream_parent_pred.reshape(-1).float().cpu()[active]
        else:
            ds = torch.full((int(active.sum()),), float("nan"))

        if image_id not in self._store:
            self._store[image_id] = {
                "reps": {k: [] for k in flat_reps},
                "y": [],
                "n": [],
                "downstream": [],
            }
        store = self._store[image_id]
        for k, v in flat_reps.items():
            store["reps"][k].append(v)
        store["y"].append(y_active)
        store["n"].append(n_active)
        store["downstream"].append(ds)

    def finalize(self, rep_names: Sequence[str] | None = None) -> Dict:
        """Build final dataset with uniform per-image subsampling.

        Returns:
            reps:     dict[str, Tensor [N_total, D]]
            y:        Tensor [N_total, 4]
            image_ids: Tensor [N_total]
            downstream_parent_pred: Tensor [N_total]
            per_image_cell_counts: dict[int, int]
        """
        if not self._store:
            raise RuntimeError("No active cells were collected")

        all_reps: dict[str, list[torch.Tensor]] = {}
        all_y: list[torch.Tensor] = []
        all_ids: list[torch.Tensor] = []
        all_ds: list[torch.Tensor] = []
        per_image_counts: dict[int, int] = {}

        for image_id, store in sorted(self._store.items()):
            # Concatenate all crops for this image
            y_img = torch.cat(store["y"], dim=0)
            ds_img = torch.cat(store["downstream"], dim=0)
            reps_img = {k: torch.cat(v, dim=0) for k, v in store["reps"].items()}

            total = y_img.shape[0]
            if total > self.max_cells_per_image:
                # Deterministic subsampling: seed keyed on image_id so result is
                # independent of insertion order
                g = torch.Generator().manual_seed(self.seed ^ (image_id * 2654435761 & 0xFFFFFFFF))
                idx = torch.randperm(total, generator=g)[: self.max_cells_per_image]
            else:
                idx = torch.arange(total)

            n_kept = int(idx.numel())
            per_image_counts[image_id] = n_kept

            y_kept = y_img[idx]
            ds_kept = ds_img[idx]
            ids_kept = torch.full((n_kept,), int(image_id), dtype=torch.long)

            all_y.append(y_kept)
            all_ids.append(ids_kept)
            all_ds.append(ds_kept)
            for k in reps_img:
                if k not in all_reps:
                    all_reps[k] = []
                all_reps[k].append(reps_img[k][idx])

        if rep_names is not None:
            # Verify all expected names exist
            missing = set(rep_names) - set(all_reps)
            if missing:
                raise ValueError(f"Missing representations after finalize: {missing}")

        return {
            "reps": {k: torch.cat(v, dim=0) for k, v in all_reps.items()},
            "y": torch.cat(all_y, dim=0),
            "image_ids": torch.cat(all_ids, dim=0),
            "downstream_parent_pred": torch.cat(all_ds, dim=0),
            "per_image_cell_counts": per_image_counts,
            "n_images": len(self._store),
        }


# ---------------------------------------------------------------------------
# Paired multi-seed MLP probe (fixes v2 unpaired seeds)
# ---------------------------------------------------------------------------

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

    The ``seed`` argument is shared across all representations by the caller
    to ensure a paired comparison (same initialisation, same mini-batch order).
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


def paired_seed_mlp_eval(
    train_reps: Dict[str, torch.Tensor],
    y_train: torch.Tensor,
    val_reps: Dict[str, torch.Tensor],
    y_val: torch.Tensor,
    image_ids_val: torch.Tensor,
    seeds: Sequence[int] = (42, 123, 456, 789, 2024),
    hidden: int = 64,
    epochs: int = 40,
    device: str | torch.device = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Evaluate primary representations with paired multi-seed MLP probes.

    All representations receive IDENTICAL seeds in IDENTICAL order, making
    the comparison paired. Results are averaged over seeds.

    Returns: dict[rep_name → image_weighted_metrics]
    """
    # Only the four primary controls are evaluated via MLP
    primary_reps = [k for k in ("op_pre", "op_post", "s2d_lossless", "s2d_pca_matched") if k in train_reps]

    seed_results: Dict[str, List[Dict[str, float]]] = {rep: [] for rep in primary_reps}
    n2plus_mask = y_val.sum(dim=1) >= 2

    for seed in seeds:
        for rep_name in primary_reps:
            pred = fit_predict_mlp(
                train_reps[rep_name],
                y_train,
                val_reps[rep_name],
                hidden=hidden,
                epochs=epochs,
                seed=seed,  # same seed for every rep at this iteration
                device=device,
            )
            m = compute_image_weighted_metrics(pred, y_val, image_ids_val)
            m_n2p = compute_image_weighted_metrics(pred, y_val, image_ids_val, mask=n2plus_mask)
            seed_results[rep_name].append({
                "parent_mae_iw": m["parent_mae_iw"],
                "composition_l1_iw": m["composition_l1_iw"],
                "n2p_parent_mae_iw": m_n2p["parent_mae_iw"],
                "n2p_composition_l1_iw": m_n2p["composition_l1_iw"],
            })

    def _avg(dicts: List[Dict[str, float]]) -> Dict[str, float]:
        keys = dicts[0].keys()
        return {k: float(sum(d[k] for d in dicts) / len(dicts)) for k in keys}

    return {rep: _avg(seed_results[rep]) for rep in primary_reps}
