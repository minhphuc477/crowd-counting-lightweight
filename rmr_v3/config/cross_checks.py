from __future__ import annotations

from typing import Any


def validate_cross_sections(cfg: dict[str, Any]) -> None:
    """Validate cross-section invariants and boolean types for solver/regional-head/Radon-OT flags."""
    # Cross-section consistency: Solver requires trained regional evidence when both sections are present
    if "model" in cfg and "loss" in cfg:
        m_cfg, l_cfg = cfg.get("model", {}), cfg.get("loss", {})
        solver_active = (
            isinstance(m_cfg, dict) and isinstance(l_cfg, dict)
            and m_cfg.get("enable_solver", False) is True and len(l_cfg) > 0
        )
        if solver_active:
            allow_unsupervised = bool(
                m_cfg.get("allow_unsupervised_solver", False)
                or cfg.get("train", {}).get("allow_unsupervised_solver", False)
            )
            has_regional_supervision = (
                float(l_cfg.get("lambda_region_nb", 0.0)) > 0.0
                or float(l_cfg.get("lambda_trunc_nb", 0.0)) > 0.0
            )
            if not has_regional_supervision and not allow_unsupervised:
                raise ValueError(
                    "Invalid configuration: model.enable_solver=True requires "
                    "loss.lambda_region_nb > 0.0 or loss.lambda_trunc_nb > 0.0. "
                    "When both are <= 0.0, the regional head is never trained. "
                    "Set model.enable_solver=False for standalone carrier, "
                    "or set model.allow_unsupervised_solver=True for unconstrained ablation."
                )

    m_cfg = cfg.get("model", {})
    if isinstance(m_cfg, dict) and "enable_regional_head" in m_cfg:
        if not isinstance(m_cfg["enable_regional_head"], bool):
            raise ValueError(
                f"model.enable_regional_head must be a boolean, got {type(m_cfg['enable_regional_head']).__name__}"
            )
        if m_cfg["enable_regional_head"] is False:
            if m_cfg.get("enable_solver", True) is not False and str(m_cfg.get("solver_mode", "")) != "direct":
                raise ValueError(
                    "model.enable_regional_head=False requires model.enable_solver=False (direct mode)."
                )
            l_cfg_chk = cfg.get("loss", {})
            if isinstance(l_cfg_chk, dict):
                for rk in ("lambda_region_nb", "lambda_trunc_nb", "lambda_region_count", "lambda_hurdle"):
                    if float(l_cfg_chk.get(rk, 0.0)) > 0.0:
                        raise ValueError(
                            f"model.enable_regional_head=False cannot be combined with loss.{rk} > 0.0."
                        )

    l_cfg = cfg.get("loss", {})
    if isinstance(l_cfg, dict) and "radon_ot_scale_by_length" in l_cfg:
        if not isinstance(l_cfg["radon_ot_scale_by_length"], bool):
            raise ValueError(
                f"loss.radon_ot_scale_by_length must be a boolean, got {type(l_cfg['radon_ot_scale_by_length']).__name__}"
            )
