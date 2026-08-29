"""Centralized model factory for HPCLite/NTPC models."""

from __future__ import annotations

from typing import Any, Dict

from .hpc_lite import HPCLite


def build_model_from_config(cfg: Dict[str, Any]) -> HPCLite:
    """Construct an HPCLite instance faithfully from a config dictionary."""
    model_cfg = cfg.get("model", cfg)
    return HPCLite(
        backbone_name=model_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=bool(model_cfg.get("pretrained", False)),
        neck_width=int(model_cfg.get("neck_width", 32)),
        context_dilations=tuple(int(d) for d in model_cfg.get("context_dilations", (1, 2, 3))),
        use_p8_context=bool(model_cfg.get("use_p8_context", False)),
        use_repblock=bool(model_cfg.get("use_repblock", False)),
        eps_d=float(model_cfg.get("eps_d", 1e-8)),
        output_stride=int(model_cfg.get("output_stride", 4)),
    )
