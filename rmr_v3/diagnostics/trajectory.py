from __future__ import annotations

from typing import Any
import torch

from rmr_core.operators import regional_sum
from .rows import resolve_scale_map


@torch.no_grad()
def compute_solver_trajectory_diagnostics(
    outputs: dict,
    target_y: torch.Tensor,
    scale_map: dict[int, int | str] | None = None,
) -> dict[str, Any]:
    """Compute trajectory MAE over iterates Y_0 -> Y_1 -> Y_2, regional disagreement,

    energy monotonicity, and harmful-correction rate.
    """
    iterates = outputs.get("iterates", [])
    if not iterates:
        return {}

    regions = outputs["regions"]
    boxes = regions.boxes
    b_region = outputs["b_region"].float()
    pred_field = outputs.get("y", outputs.get("y0"))
    target_float = target_y.float()
    if pred_field is not None and target_float.shape[-2:] != pred_field.shape[-2:]:
        target_float = target_float[..., :pred_field.shape[-2], :pred_field.shape[-1]]

    gt_reg = regional_sum(target_float, boxes, out_dtype=torch.float32)

    scale_ids = regions.scale_id
    resolved_scale_map = resolve_scale_map(scale_ids, scale_map)

    results: dict[str, Any] = {}

    for t_idx, y_t in enumerate(iterates):
        reg_t = regional_sum(y_t.float(), boxes, out_dtype=torch.float32)
        abs_err = (reg_t - gt_reg).abs()
        results[f"mae_reg_y{t_idx}"] = float(abs_err.mean().item())

        for sid, s_px in resolved_scale_map.items():
            mask = scale_ids == sid
            if mask.any():
                results[f"mae_reg_{s_px}_y{t_idx}"] = float(abs_err[..., mask].mean().item())
            else:
                results[f"mae_reg_{s_px}_y{t_idx}"] = 0.0

        # Disagreement with predicted regional evidence mu
        disagree = (reg_t - b_region).abs()
        results[f"reg_disagreement_y{t_idx}"] = float(disagree.mean().item())

    # Energy monotonicity
    energy_trace = outputs.get("energy_trace", [])
    if energy_trace:
        mono_count = 0
        total_transitions = len(energy_trace)
        for step in energy_trace:
            e_before = float(step["before"].mean().item())
            e_after = float(step["after"].mean().item())
            if e_after <= e_before + 1e-8:
                mono_count += 1
        results["energy_monotonic_fraction"] = float(mono_count / max(1, total_transitions))
    else:
        results["energy_monotonic_fraction"] = 1.0

    # Image-level harmful correction rate: e_0 = |sum Y_0 - N*|, e_T = |sum Y_T - N*|
    y0 = iterates[0]
    y_final = iterates[-1]

    cnt0 = y0.sum(dim=(-1, -2, -3)).float().cpu()
    cnt_final = y_final.sum(dim=(-1, -2, -3)).float().cpu()
    gt_cnt = target_float.sum(dim=(-1, -2, -3)).float().cpu()

    e0 = (cnt0 - gt_cnt).abs()
    e_final = (cnt_final - gt_cnt).abs()
    delta_e = e_final - e0  # < 0 means solver helped, > 0 means solver hurt

    help_mask = delta_e < -1e-4
    harm_mask = delta_e > 1e-4
    neutral_mask = ~help_mask & ~harm_mask

    results["solver_help_fraction"] = float(help_mask.float().mean().item())
    results["solver_harm_fraction"] = float(harm_mask.float().mean().item())
    results["solver_neutral_fraction"] = float(neutral_mask.float().mean().item())
    results["solver_delta_e_mean"] = float(delta_e.mean().item())

    return results
