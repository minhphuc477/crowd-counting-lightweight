"""Centralized model factory and checkpoint compatibility validator for HPCLite/NTPC."""

from __future__ import annotations

from typing import Any, Dict

from .hpc_lite import HPCLite

_KNOWN_MODEL_KEYS = {
    "backbone",
    "pretrained",
    "neck_width",
    "context_dilations",
    "use_p8_context",
    "use_repblock",
    "eps_d",
    "output_stride",
    "init_checkpoint",
}


def _parse_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Cannot parse boolean from string '{val}'")
    raise TypeError(f"Invalid type for boolean: {type(val)}")


def build_model_from_config(cfg: Dict[str, Any]) -> HPCLite:
    """Construct an HPCLite instance faithfully from a config dictionary."""
    model_cfg = cfg.get("model", cfg)
    unknown_keys = set(model_cfg.keys()) - _KNOWN_MODEL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown keys in model config: {sorted(unknown_keys)}")

    return HPCLite(
        backbone_name=str(model_cfg.get("backbone", "mobilenetv4_conv_small_050")),
        pretrained=_parse_bool(model_cfg.get("pretrained", False)),
        neck_width=int(model_cfg.get("neck_width", 32)),
        context_dilations=tuple(int(d) for d in model_cfg.get("context_dilations", (1, 2, 3))),
        use_p8_context=_parse_bool(model_cfg.get("use_p8_context", False)),
        use_repblock=_parse_bool(model_cfg.get("use_repblock", False)),
        eps_d=float(model_cfg.get("eps_d", 1e-8)),
        output_stride=int(model_cfg.get("output_stride", 4)),
    )


def assert_checkpoint_compatible(checkpoint: dict, cfg: dict) -> None:
    """Assert that the checkpoint's embedded configuration matches the active YAML config."""
    trained = checkpoint.get("config")
    if trained is None or not isinstance(trained, dict):
        return

    old_model = trained.get("model", {})
    new_model = cfg.get("model", {})
    for k in (
        "backbone",
        "neck_width",
        "context_dilations",
        "use_p8_context",
        "use_repblock",
        "eps_d",
        "output_stride",
    ):
        if k in old_model and k in new_model:
            old_v, new_v = old_model[k], new_model[k]
            if k in {"use_p8_context", "use_repblock", "pretrained"}:
                old_v, new_v = _parse_bool(old_v), _parse_bool(new_v)
            elif k in {"eps_d"}:
                old_v, new_v = float(old_v), float(new_v)
            elif k in {"context_dilations"}:
                old_v, new_v = list(old_v), list(new_v)
            if old_v != new_v:
                raise ValueError(
                    f"Model config mismatch for '{k}': checkpoint has {old_v!r}, active config has {new_v!r}"
                )

    ds_keys = ("name", "part", "coordinate_base", "image_mean", "image_std")
    old_ds = trained.get("dataset", {})
    new_ds = cfg.get("dataset", {})

    mismatches = {}
    for key in ds_keys:
        if key in old_ds and key in new_ds and old_ds[key] != new_ds[key]:
            mismatches[key] = (old_ds[key], new_ds[key])

    if mismatches:
        raise ValueError(
            f"Dataset/preprocessing config mismatch between checkpoint and active YAML: {mismatches}"
        )

