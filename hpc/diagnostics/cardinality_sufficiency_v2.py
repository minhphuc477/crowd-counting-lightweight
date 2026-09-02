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
