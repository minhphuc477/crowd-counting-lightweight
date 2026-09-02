from __future__ import annotations

"""Objective Mechanism Audit v2: Flat-DM16 vs Hierarchical DTM Tree.

Investigates why Flat-DM16 (R2) strongly outperforms Hierarchical DTM Tree (R4/R5)
in dense crowd regimes and exhibits near-zero bias (-1.85 vs -21.94).

Evaluates on frozen model mass predictions and exact ground-truth count pyramids:
1. Fixed total gradient: computed directly from criterion(mass) total loss, not sum of logs.
2. Euler scale projection: <g, m> and cos(g, m) (true scale-invariance check for DM).
3. Parameter-space count direction:
   Delta_N_k approx -eta <grad_theta(N), grad_theta(L_k)>
   - inner product > 0 => loss step decreases predicted count in model weight space.
   - inner product < 0 => loss step increases predicted count.
4. Actual cancellation ratio:
   C(g_a, g_b) = 1 - ||g_a + g_b|| / (||g_a|| + ||g_b||) in [0, 1].
5. Full pairwise tree gradient matrix cos(g_a, g_b) and cancellation C.
6. Evaluated on BOTH R2 checkpoint and R4 checkpoint.
7. Density stratification by local crop count quantiles: Low (<50), Med (50-150), High (>=150).
8. Kappa sweep across {2, 5, 10, 20, 50, 100} tracking tree cancellation C and cos(64->32, 32->16).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hpc.losses.ntpc import NTPCConfig, NTPCLoss


def cancellation_ratio(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute gradient cancellation ratio C = 1 - ||g1 + g2|| / (||g1|| + ||g2||).

    Returns value in [0, 1]:
    - C = 0: perfectly aligned / zero cancellation.
    - C ~ 0.293: orthogonal.
    - C = 1.0: perfectly anti-aligned / 100% destructive cancellation.
    """
    v1 = g1.detach().flatten().double()
    v2 = g2.detach().flatten().double()
    n1 = float(v1.norm().item())
    n2 = float(v2.norm().item())
    if n1 <= eps or n2 <= eps:
        return 0.0
    sum_norm = float((v1 + v2).norm().item())
    denom = n1 + n2
    ratio = max(0.0, min(1.0, 1.0 - (sum_norm / denom)))
    return float(ratio)


def compute_pairwise_cosine(
    g1: torch.Tensor,
    g2: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Cosine similarity between two flattened vectors."""
    v1 = g1.detach().flatten().double()
    v2 = g2.detach().flatten().double()
    n1 = float(v1.norm().item())
    n2 = float(v2.norm().item())
    if n1 <= eps or n2 <= eps:
        return float("nan")
    cos = float(torch.dot(v1, v2).item() / (n1 * n2))
    return max(-1.0, min(1.0, cos))


def compute_component_gradients(
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    criterion: NTPCLoss,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Compute exact per-component gradients d(L_k) / d(mass) and true total gradient.

    Fixes bug in v1: g_total is computed directly from criterion's returned total loss,
    avoiding duplicate root loss summation.
    """
    if mass.ndim != 4 or mass.shape[0] != 1 or mass.shape[1] != 1:
        raise ValueError(f"Expected mass shape [1, 1, H, W], got {tuple(mass.shape)}")

    mass_leaf = mass.detach().float().requires_grad_(True)
    total_loss, _, components = criterion(
        mass_leaf, targets, return_components=True, validate_targets=False
    )

    grads: Dict[str, torch.Tensor] = {}
    for name, comp_loss in components.items():
        if not comp_loss.requires_grad or comp_loss.grad_fn is None:
            grads[name] = torch.zeros_like(mass_leaf)
            continue
        g = torch.autograd.grad(comp_loss, mass_leaf, retain_graph=True)[0]
        grads[name] = g.detach()

    # True total gradient directly from total_loss (avoids double-counting root)
    g_total = torch.autograd.grad(total_loss, mass_leaf, retain_graph=False)[0]

    return grads, g_total.detach()


def compute_mass_gradient_metrics(
    grad: torch.Tensor,
    mass: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute directional metrics for a mass-map gradient g = d(L)/d(mass).

    1. norm: ||g||_2
    2. grad_sum: sum(g)
    3. count_push: -sum(g) (mass additive push)
    4. euler_dot: <g, m> (inner product with current mass map)
    5. euler_cos: cos(g, m) = <g, m> / (||g|| ||m||) (scale-invariance alignment)
    6. uniform_cos: cos(g, 1) = sum(g) / (sqrt(N) * ||g||) (additive perturbation alignment)
    """
    g = grad.detach().flatten().double()
    m = mass.detach().flatten().double()
    norm_g = float(g.norm().item())
    norm_m = float(m.norm().item())
    grad_sum = float(g.sum().item())
    n_elem = g.numel()

    if norm_g <= eps or n_elem == 0:
        return {
            "norm": 0.0,
            "grad_sum": 0.0,
            "count_push": 0.0,
            "euler_dot": 0.0,
            "euler_cos": 0.0,
            "uniform_cos": 0.0,
        }

    euler_dot = float(torch.dot(g, m).item())
    euler_cos = euler_dot / (norm_g * norm_m + eps) if norm_m > eps else 0.0
    uniform_cos = grad_sum / (math.sqrt(n_elem) * norm_g)

    return {
        "norm": norm_g,
        "grad_sum": grad_sum,
        "count_push": -grad_sum,
        "euler_dot": euler_dot,
        "euler_cos": float(max(-1.0, min(1.0, euler_cos))),
        "uniform_cos": float(max(-1.0, min(1.0, uniform_cos))),
    }


def compute_parameter_space_metrics(
    model: nn.Module,
    crop_img: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    criterion: NTPCLoss,
    active_components: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute parameter-space count direction and loss gradients.

    Evaluates:
      v_N = grad_theta(N_pred)
      v_k = grad_theta(L_k)
      Delta_N_k approx -eta <v_N, v_k>
      cos_param = <v_N, v_k> / (||v_N|| * ||v_k||)
      - cos_param > 0 => loss step decreases predicted count in model parameter space!
      - cos_param < 0 => loss step increases predicted count.
    """
    model.zero_grad(set_to_none=True)
    mass = model.forward_mass(crop_img)
    pred_count = mass.sum()

    # 1. Parameter gradient of predicted count
    grad_N = torch.autograd.grad(pred_count, model.parameters(), retain_graph=True)
    v_N = torch.cat([g.detach().flatten().double() for g in grad_N])
    norm_N = float(v_N.norm().item())

    # 2. Forward criterion
    total_loss, _, components = criterion(mass, targets, return_components=True, validate_targets=False)

    param_metrics: Dict[str, Dict[str, float]] = {}
    eps = 1e-12

    for name in active_components:
        if name not in components or not components[name].requires_grad or components[name].grad_fn is None:
            continue
        grad_k = torch.autograd.grad(components[name], model.parameters(), retain_graph=True)
        v_k = torch.cat([g.detach().flatten().double() for g in grad_k])
        norm_k = float(v_k.norm().item())
        if norm_k <= eps or norm_N <= eps:
            param_metrics[name] = {
                "norm_theta": norm_k,
                "count_dot_theta": 0.0,
                "count_cos_theta": 0.0,
            }
        else:
            dot_theta = float(torch.dot(v_N, v_k).item())
            cos_theta = dot_theta / (norm_N * norm_k)
            param_metrics[name] = {
                "norm_theta": norm_k,
                "count_dot_theta": dot_theta,
                "count_cos_theta": float(max(-1.0, min(1.0, cos_theta))),
            }

    # Total loss parameter gradient
    grad_tot = torch.autograd.grad(total_loss, model.parameters(), retain_graph=False)
    v_tot = torch.cat([g.detach().flatten().double() for g in grad_tot])
    norm_tot = float(v_tot.norm().item())
    if norm_tot > eps and norm_N > eps:
        dot_tot = float(torch.dot(v_N, v_tot).item())
        cos_tot = dot_tot / (norm_N * norm_tot)
        param_metrics["total"] = {
            "norm_theta": norm_tot,
            "count_dot_theta": dot_tot,
            "count_cos_theta": float(max(-1.0, min(1.0, cos_tot))),
        }

    return param_metrics


def compute_audit_for_mode_v2(
    model: Optional[nn.Module],
    crop_img: Optional[torch.Tensor],
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    criterion: NTPCLoss,
    active_components: Sequence[str],
) -> Dict[str, Any]:
    """Audit one objective mode on a single crop with both mass-space and parameter-space metrics."""
    grads, g_total = compute_component_gradients(mass, targets, criterion)

    # Mass-space metrics
    component_metrics = {
        name: compute_mass_gradient_metrics(grads[name], mass)
        for name in active_components
        if name in grads
    }
    component_metrics["total"] = compute_mass_gradient_metrics(g_total, mass)

    # Pairwise cosines and cancellations among active components
    cosines: Dict[str, float] = {}
    cancellations: Dict[str, float] = {}
    valid_names = [n for n in active_components if n in grads]
    for i, name_a in enumerate(valid_names):
        for j, name_b in enumerate(valid_names):
            if i < j:
                pair_key = f"{name_a}_vs_{name_b}"
                cosines[pair_key] = compute_pairwise_cosine(grads[name_a], grads[name_b])
                cancellations[pair_key] = cancellation_ratio(grads[name_a], grads[name_b])

    # Parameter-space metrics (if model and crop_img provided)
    param_metrics = None
    if model is not None and crop_img is not None:
        param_metrics = compute_parameter_space_metrics(
            model, crop_img, targets, criterion, active_components
        )

    return {
        "component_metrics": component_metrics,
        "pairwise_cosines": cosines,
        "pairwise_cancellations": cancellations,
        "param_metrics": param_metrics,
        "raw_grads": {k: grads[k] for k in valid_names},
        "total_grad": g_total,
    }


def sweep_kappa_on_crop_v2(
    mass: torch.Tensor,
    targets: Dict[int | str, torch.Tensor],
    kappas: Sequence[float] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Sweep kappa tracking tree cancellation C and cos(64->32, 32->16)."""
    results_r2: Dict[str, Any] = {}
    results_r4: Dict[str, Any] = {}

    for kappa in kappas:
        k_str = f"k_{int(kappa) if kappa == int(kappa) else kappa}"

        # R2 Flat-DM16
        cfg_r2 = NTPCConfig(mode="r2_flat_dm", root_loss="nb", kappa_flat16=float(kappa))
        crit_r2 = NTPCLoss(cfg_r2).to(device)
        audit_r2 = compute_audit_for_mode_v2(
            None, None, mass, targets, crit_r2, active_components=("root_magnitude", "flat_16")
        )
        results_r2[k_str] = {
            "flat_16": audit_r2["component_metrics"]["flat_16"],
            "total": audit_r2["component_metrics"]["total"],
            "cos_root_flat": audit_r2["pairwise_cosines"].get("root_magnitude_vs_flat_16", float("nan")),
            "cancellation_root_flat": audit_r2["pairwise_cancellations"].get("root_magnitude_vs_flat_16", float("nan")),
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
        audit_r4 = compute_audit_for_mode_v2(
            None, None, mass, targets, crit_r4,
            active_components=("root_magnitude", "root_to_64", "64_to_32", "32_to_16"),
        )
        results_r4[k_str] = {
            "root_to_64": audit_r4["component_metrics"]["root_to_64"],
            "64_to_32": audit_r4["component_metrics"]["64_to_32"],
            "32_to_16": audit_r4["component_metrics"]["32_to_16"],
            "total": audit_r4["component_metrics"]["total"],
            "cos_64_32_vs_32_16": audit_r4["pairwise_cosines"].get("64_to_32_vs_32_to_16", float("nan")),
            "cancellation_64_32_vs_32_16": audit_r4["pairwise_cancellations"].get("64_to_32_vs_32_to_16", float("nan")),
            "cos_root_vs_32_16": audit_r4["pairwise_cosines"].get("root_magnitude_vs_32_to_16", float("nan")),
        }

    return {"r2_flat": results_r2, "r4_tree": results_r4}


def stratify_by_local_crop_count(
    records: List[Dict[str, Any]],
    threshold_low: float = 50.0,
    threshold_high: float = 150.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Stratify audit records by local crop count (Low: <50, Medium: 50-150, High: >=150)."""
    bins: Dict[str, List[Dict[str, Any]]] = {
        "all": records,
        "low (<50)": [],
        "medium (50-150)": [],
        "high (>=150)": [],
    }
    for r in records:
        gt_n = float(r["gt_count"])
        if gt_n < threshold_low:
            bins["low (<50)"].append(r)
        elif gt_n < threshold_high:
            bins["medium (50-150)"].append(r)
        else:
            bins["high (>=150)"].append(r)
    return bins


def summarize_audit_group_v2(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics for a group of audit records."""
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

    return {
        "count": n,
        "mean_gt_count": _mean(["gt_count"]),
        "mean_pred_count": _mean(["pred_count"]),
        "mean_count_error": _mean(["signed_error"]),
        # R2
        "r2": {
            "flat16_norm": _mean(["r2", "component_metrics", "flat_16", "norm"]),
            "flat16_euler_cos": _mean(["r2", "component_metrics", "flat_16", "euler_cos"]),
            "flat16_param_cos": _mean(["r2", "param_metrics", "flat_16", "count_cos_theta"]),
            "root_param_cos": _mean(["r2", "param_metrics", "root_magnitude", "count_cos_theta"]),
            "total_param_cos": _mean(["r2", "param_metrics", "total", "count_cos_theta"]),
            "cos_root_vs_flat16": _mean(["r2", "pairwise_cosines", "root_magnitude_vs_flat_16"]),
            "conflict_root_vs_flat16": _conflict_rate(["r2", "pairwise_cosines", "root_magnitude_vs_flat_16"]),
            "cancellation_root_vs_flat16": _mean(["r2", "pairwise_cancellations", "root_magnitude_vs_flat_16"]),
        },
        # R4
        "r4": {
            "32_16_norm": _mean(["r4", "component_metrics", "32_to_16", "norm"]),
            "64_32_norm": _mean(["r4", "component_metrics", "64_to_32", "norm"]),
            "32_16_euler_cos": _mean(["r4", "component_metrics", "32_to_16", "euler_cos"]),
            "64_32_euler_cos": _mean(["r4", "component_metrics", "64_to_32", "euler_cos"]),
            "32_16_param_cos": _mean(["r4", "param_metrics", "32_to_16", "count_cos_theta"]),
            "64_32_param_cos": _mean(["r4", "param_metrics", "64_to_32", "count_cos_theta"]),
            "root_param_cos": _mean(["r4", "param_metrics", "root_magnitude", "count_cos_theta"]),
            "total_param_cos": _mean(["r4", "param_metrics", "total", "count_cos_theta"]),
            "cos_64_32_vs_32_16": _mean(["r4", "pairwise_cosines", "64_to_32_vs_32_to_16"]),
            "conflict_64_32_vs_32_16": _conflict_rate(["r4", "pairwise_cosines", "64_to_32_vs_32_to_16"]),
            "cancellation_64_32_vs_32_16": _mean(["r4", "pairwise_cancellations", "64_to_32_vs_32_to_16"]),
            "cos_root_vs_32_16": _mean(["r4", "pairwise_cosines", "root_magnitude_vs_32_to_16"]),
            "conflict_root_vs_32_16": _conflict_rate(["r4", "pairwise_cosines", "root_magnitude_vs_32_to_16"]),
        },
    }
