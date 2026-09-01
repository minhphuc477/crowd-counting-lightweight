"""Central factory for NTPC criteria to eliminate duplication and configuration drift."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .ntpc import NTPCConfig, NTPCLoss


def build_ntpc_criterion_from_config(
    cfg: Mapping[str, Any],
    *,
    crop_statistics: Optional[Mapping[str, Any]] = None,
) -> NTPCLoss:
    """Construct an NTPCLoss criterion from configuration and resolved crop statistics."""
    loss_cfg = cfg.get("loss", {})
    stats_cfg = cfg.get("statistics", {})
    shared_kappa = loss_cfg.get("kappa_shared")

    def kappa(name: str, default: float = 20.0) -> float:
        if name in loss_cfg:
            return float(loss_cfg[name])
        if shared_kappa is not None:
            return float(shared_kappa)
        return float(default)

    threshold_value = loss_cfg.get("dense_threshold_16", "auto")
    if threshold_value is None or str(threshold_value).lower() == "auto":
        if crop_statistics is None:
            mode = loss_cfg.get("mode", "r4_dtm_tree16")
            if "r5" in mode:
                raise ValueError("dense_threshold_16=auto requires resolved crop statistics")
            dense_threshold = 0.0
        else:
            dense_threshold = float(crop_statistics.get("dense_threshold_q85", 0.0))
    else:
        dense_threshold = float(threshold_value)

    config = NTPCConfig(
        mode=loss_cfg.get("mode", "r4_dtm_tree16"),
        root_loss=loss_cfg.get("root_loss", "nb"),
        root_dispersion=float(stats_cfg.get("root_dispersion", 50.0)),
        kappa_root64=kappa("kappa_root64"),
        kappa_64_32=kappa("kappa_64_32"),
        kappa_32_16=kappa("kappa_32_16"),
        kappa_16_8=kappa("kappa_16_8"),
        kappa_8_4=kappa("kappa_8_4"),
        kappa_flat16=kappa("kappa_flat16"),
        dense_threshold_16=dense_threshold,
        w_root_nb=float(loss_cfg.get("w_root_nb", 1.0)),
        w_root64=float(loss_cfg.get("w_root64", 1.0)),
        w_64_32=float(loss_cfg.get("w_64_32", 1.0)),
        w_32_16=float(loss_cfg.get("w_32_16", 1.0)),
        w_16_8=float(loss_cfg.get("w_16_8", 1.0)),
        w_8_4=float(loss_cfg.get("w_8_4", 1.0)),
        w_flat_16=float(loss_cfg.get("w_flat_16", 1.0)),
        w_exact_regression=float(loss_cfg.get("w_exact_regression", 1.0)),
        w_deterministic_alloc=float(loss_cfg.get("w_deterministic_alloc", 1.0)),
        eps=float(loss_cfg.get("eps", 1e-8)),
    )
    return NTPCLoss(config)
