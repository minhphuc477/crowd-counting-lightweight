from __future__ import annotations

"""Objective Mechanism Audit: Flat-DM16 vs Hierarchical DTM Tree.

Investigates why Flat-DM16 (R2) strongly outperforms Hierarchical DTM Tree (R4/R5)
in dense crowd regimes and exhibits near-zero bias (-1.85 vs -21.94).

Evaluates on frozen model mass predictions and exact ground-truth count pyramids:
1. Exact component gradients g_k = d(L_k) / d(mass) at output stride 4.
2. Directional magnitude cosine rho_magnitude = cos(g_k, 1):
   - rho > 0: loss component actively pushes count DOWN (negative bias / under-counting pressure).
   - rho < 0: loss component pushes count UP (over-counting pressure).
   - rho = 0: mass-neutral pure spatial allocation.
3. Pairwise gradient cosine conflict cos(g_a, g_b) between loss components.
4. Density-stratified analysis: Sparse (N < 300), Medium (300 <= N < 1000), Dense (N >= 1000).
5. Offline kappa sweep across {2, 5, 10, 20, 50, 100}.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from hpc.losses.ntpc import NTPCConfig, NTPCLoss


def compute_component_gradients(
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    criterion: NTPCLoss,
) -> Dict[str, torch.Tensor]:
    """Compute exact per-component gradients d(L_k) / d(mass) on a single crop.

    Args:
        mass: [1, 1, H, W] positive predicted count mass map at stride 4.
        targets: exact integer count pyramid dict (4, 8, 16, 32, 64, "N").
        criterion: NTPCLoss instance (e.g. mode='r2_flat_dm' or 'r4_dtm_tree16').

    Returns:
        Dict mapping component name -> gradient tensor [1, 1, H, W].
    """
    if mass.ndim != 4 or mass.shape[0] != 1 or mass.shape[1] != 1:
        raise ValueError(f"Expected mass shape [1, 1, H, W], got {tuple(mass.shape)}")

    mass_leaf = mass.detach().float().requires_grad_(True)
    _, _, components = criterion(
        mass_leaf, targets, return_components=True, validate_targets=False
    )

    grads: Dict[str, torch.Tensor] = {}
    for name, comp_loss in components.items():
        if not comp_loss.requires_grad or comp_loss.grad_fn is None:
            # Constant or zero term
            grads[name] = torch.zeros_like(mass_leaf)
            continue
        g = torch.autograd.grad(comp_loss, mass_leaf, retain_graph=True)[0]
        grads[name] = g.detach()

    # Also compute total gradient
    total_loss = sum(components.values())
    g_total = torch.autograd.grad(total_loss, mass_leaf, retain_graph=False)[0]
    grads["total"] = g_total.detach()

    return grads


def compute_gradient_metrics(
    grad: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute directional magnitude cosine and scale metrics for a single gradient map.

    rho_magnitude = cos(grad, 1) = sum(grad) / (sqrt(numel) * ||grad||_2)
    - rho > 0: gradient aligns with positive mass direction -> gradient descent
      step (-grad) pushes mass DOWN (negative magnitude bias / undercounting).
    - rho < 0: gradient descent step pushes mass UP (overcounting).
    - rho == 0: purely mass-neutral spatial redistribution.
    """
    g = grad.detach().flatten().double()
    norm = float(g.norm().item())
    grad_sum = float(g.sum().item())
    n_elem = g.numel()

    if norm <= eps or n_elem == 0:
        return {
            "norm": 0.0,
            "grad_sum": 0.0,
            "count_push": 0.0,
            "magnitude_cosine": 0.0,
        }

    # cos(g, 1) = sum(g) / (sqrt(N) * ||g||)
    mag_cos = grad_sum / (math.sqrt(n_elem) * norm)
    mag_cos = max(-1.0, min(1.0, mag_cos))

    return {
        "norm": norm,
        "grad_sum": grad_sum,
        "count_push": -grad_sum,  # -grad direction: positive means pushing count UP
        "magnitude_cosine": float(mag_cos),
    }


def compute_pairwise_cosine(
    g1: torch.Tensor,
    g2: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between two gradient maps."""
    v1 = g1.detach().flatten().double()
    v2 = g2.detach().flatten().double()
    n1 = float(v1.norm().item())
    n2 = float(v2.norm().item())
    if n1 <= eps or n2 <= eps:
        return float("nan")
    cos = float(torch.dot(v1, v2).item() / (n1 * n2))
    return max(-1.0, min(1.0, cos))


def compute_audit_for_mode(
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    criterion: NTPCLoss,
    active_components: Sequence[str],
) -> Dict[str, Any]:
    """Audit one objective mode on a single crop.

    Returns metrics for each component, pairwise cosine similarities, and total gradient.
    """
    grads = compute_component_gradients(mass, targets, criterion)

    component_metrics = {
        name: compute_gradient_metrics(grads[name])
        for name in active_components
        if name in grads
    }
    component_metrics["total"] = compute_gradient_metrics(grads["total"])

    # Pairwise cosines among active components
    cosines: Dict[str, float] = {}
    valid_names = [n for n in active_components if n in grads]
    for i, name_a in enumerate(valid_names):
        for j, name_b in enumerate(valid_names):
            if i < j:
                cos_val = compute_pairwise_cosine(grads[name_a], grads[name_b])
                cosines[f"{name_a}_vs_{name_b}"] = cos_val

    return {
        "component_metrics": component_metrics,
        "pairwise_cosines": cosines,
        "raw_grads": {k: grads[k] for k in valid_names},
        "total_grad": grads["total"],
    }


def sweep_kappa_on_crop(
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    kappas: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Evaluate R2 Flat-DM16 and R4 Tree-DM across a sweep of kappa values on the same crop."""
    results_r2: Dict[str, Any] = {}
    results_r4: Dict[str, Any] = {}

    for kappa in kappas:
        k_str = f"k_{int(kappa) if kappa == int(kappa) else kappa}"

        # R2 Flat-DM16
        cfg_r2 = NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=float(kappa))
        crit_r2 = NTPCLoss(cfg_r2).to(device)
        audit_r2 = compute_audit_for_mode(
            mass, targets, crit_r2, active_components=("root_magnitude", "flat_16")
        )
        # Store metrics without raw tensors for serializability
        results_r2[k_str] = {
            "flat_16": audit_r2["component_metrics"]["flat_16"],
            "total": audit_r2["component_metrics"]["total"],
            "cos_root_flat": audit_r2["pairwise_cosines"].get("root_magnitude_vs_flat_16", float("nan")),
        }

        # R4 Neural DTM Tree
        cfg_r4 = NTPCConfig(
            mode="r4_dtm_tree16",
            root_loss="nb",
            kappa_root64=float(kappa),
            kappa_64_32=float(kappa),
            kappa_32_16=float(kappa),
        )
        crit_r4 = NTPCLoss(cfg_r4).to(device)
        audit_r4 = compute_audit_for_mode(
            mass, targets, crit_r4,
            active_components=("root_magnitude", "root_to_64", "64_to_32", "32_to_16"),
        )
        results_r4[k_str] = {
            "root_to_64": audit_r4["component_metrics"]["root_to_64"],
            "64_to_32": audit_r4["component_metrics"]["64_to_32"],
            "32_to_16": audit_r4["component_metrics"]["32_to_16"],
            "total": audit_r4["component_metrics"]["total"],
            "cos_root_vs_32_16": audit_r4["pairwise_cosines"].get("root_magnitude_vs_32_to_16", float("nan")),
            "cos_64_32_vs_32_16": audit_r4["pairwise_cosines"].get("64_to_32_vs_32_to_16", float("nan")),
        }

    return {"r2_flat": results_r2, "r4_tree": results_r4}


def stratify_by_density(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Stratify audit records into Sparse (N < 300), Medium (300 <= N < 1000), Dense (N >= 1000)."""
    bins: Dict[str, List[Dict[str, Any]]] = {
        "all": records,
        "sparse": [],
        "medium": [],
        "dense": [],
    }
    for r in records:
        gt_n = float(r["gt_count"])
        if gt_n < 300:
            bins["sparse"].append(r)
        elif gt_n < 1000:
            bins["medium"].append(r)
        else:
            bins["dense"].append(r)
    return bins


def summarize_audit_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate means and conflict rates across a list of crop audit records."""
    if not records:
        return {"count": 0}

    n = len(records)

    def _mean(key_path: Sequence[str]) -> float:
        vals = []
        for r in records:
            curr = r
            for k in key_path:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    curr = None
                    break
            if curr is not None and isinstance(curr, (int, float)) and math.isfinite(curr):
                vals.append(float(curr))
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _conflict_rate(key_path: Sequence[str]) -> float:
        """Fraction of crops where cosine similarity < 0."""
        conflicts = 0
        valid = 0
        for r in records:
            curr = r
            for k in key_path:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    curr = None
                    break
            if curr is not None and isinstance(curr, (int, float)) and math.isfinite(curr):
                valid += 1
                if curr < 0.0:
                    conflicts += 1
        return float(conflicts / valid) if valid else float("nan")

    summary: Dict[str, Any] = {
        "count": n,
        "mean_gt_count": _mean(["gt_count"]),
        "mean_pred_count": _mean(["pred_count"]),
        "mean_count_error": _mean(["signed_error"]),
        # R2 metrics
        "r2": {
            "root_mag_cos": _mean(["r2", "component_metrics", "root_magnitude", "magnitude_cosine"]),
            "flat16_mag_cos": _mean(["r2", "component_metrics", "flat_16", "magnitude_cosine"]),
            "total_mag_cos": _mean(["r2", "component_metrics", "total", "magnitude_cosine"]),
            "flat16_grad_norm": _mean(["r2", "component_metrics", "flat_16", "norm"]),
            "root_grad_norm": _mean(["r2", "component_metrics", "root_magnitude", "norm"]),
            "cos_root_vs_flat16": _mean(["r2", "pairwise_cosines", "root_magnitude_vs_flat_16"]),
            "conflict_root_vs_flat16": _conflict_rate(["r2", "pairwise_cosines", "root_magnitude_vs_flat_16"]),
        },
        # R4 metrics
        "r4": {
            "root_mag_cos": _mean(["r4", "component_metrics", "root_magnitude", "magnitude_cosine"]),
            "root_to_64_mag_cos": _mean(["r4", "component_metrics", "root_to_64", "magnitude_cosine"]),
            "64_to_32_mag_cos": _mean(["r4", "component_metrics", "64_to_32", "magnitude_cosine"]),
            "32_to_16_mag_cos": _mean(["r4", "component_metrics", "32_to_16", "magnitude_cosine"]),
            "total_mag_cos": _mean(["r4", "component_metrics", "total", "magnitude_cosine"]),
            "32_to_16_grad_norm": _mean(["r4", "component_metrics", "32_to_16", "norm"]),
            "cos_root_vs_32_16": _mean(["r4", "pairwise_cosines", "root_magnitude_vs_32_to_16"]),
            "conflict_root_vs_32_16": _conflict_rate(["r4", "pairwise_cosines", "root_magnitude_vs_32_to_16"]),
            "cos_64_32_vs_32_16": _mean(["r4", "pairwise_cosines", "64_to_32_vs_32_to_16"]),
            "conflict_64_32_vs_32_16": _conflict_rate(["r4", "pairwise_cosines", "64_to_32_vs_32_to_16"]),
        },
    }

    return summary
