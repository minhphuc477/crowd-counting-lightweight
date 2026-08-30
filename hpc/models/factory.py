"""Centralized model factory and checkpoint compatibility validator for HPCLite/NTPC."""

from __future__ import annotations

import math
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
    "features",
}


def _parse_features(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("C4", "C8", "C16")
    if not isinstance(value, (list, tuple)):
        raise TypeError("model.features must be a list such as [C4, C8, C16, C32]")
    features = tuple(str(item).strip().upper() for item in value)
    if features not in {("C4", "C8", "C16"), ("C4", "C8", "C16", "C32")}:
        raise ValueError(
            "model.features must be [C4, C8, C16] or [C4, C8, C16, C32], "
            f"got {list(features)}"
        )
    return features


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


def resolve_model_config(cfg: dict) -> dict:
    m = cfg.get("model", cfg)
    return {
        "backbone": str(m.get("backbone", "mobilenetv4_conv_small_050")),
        "pretrained": _parse_bool(m.get("pretrained", False)),
        "neck_width": int(m.get("neck_width", 32)),
        "context_dilations": tuple(int(x) for x in m.get("context_dilations", (1, 2, 3))),
        "use_p8_context": _parse_bool(m.get("use_p8_context", False)),
        "use_repblock": _parse_bool(m.get("use_repblock", False)),
        "eps_d": float(m.get("eps_d", 1e-8)),
        "output_stride": int(m.get("output_stride", 4)),
        "features": _parse_features(m.get("features")),
    }


def resolve_dataset_config(cfg: dict) -> dict:
    d = cfg.get("dataset", {})
    return {
        "name": str(d.get("name", "")),
        "part": str(d.get("part", "part_A")),
        "coordinate_base": int(d.get("coordinate_base", 1)),
        "image_mean": tuple(float(x) for x in d.get("image_mean", [0.485, 0.456, 0.406])),
        "image_std": tuple(float(x) for x in d.get("image_std", [0.229, 0.224, 0.225])),
    }


def resolve_pretrained_spec(cfg: dict) -> dict | None:
    """Resolve immutable timm weight/input metadata without downloading weights."""
    resolved = resolve_model_config(cfg)
    if not resolved["pretrained"]:
        return None

    import timm

    try:
        pretrained_cfg = timm.get_pretrained_cfg(resolved["backbone"])
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve pretrained weights for backbone '{resolved['backbone']}'"
        ) from exc
    if pretrained_cfg is None:
        raise ValueError(f"Backbone '{resolved['backbone']}' has no timm pretrained configuration")

    source = pretrained_cfg.hf_hub_id or pretrained_cfg.url or pretrained_cfg.file
    if not source:
        raise ValueError(
            f"Backbone '{resolved['backbone']}' declares pretrained=True but has no weight source"
        )
    return {
        "architecture": pretrained_cfg.architecture,
        "tag": pretrained_cfg.tag,
        "source": str(source),
        "mean": tuple(float(x) for x in pretrained_cfg.mean),
        "std": tuple(float(x) for x in pretrained_cfg.std),
        "input_size": tuple(int(x) for x in pretrained_cfg.input_size),
        "license": pretrained_cfg.license,
    }


def validate_pretrained_normalization(cfg: dict, atol: float = 1e-8) -> dict | None:
    """Fail fast when dataset normalization does not match the selected timm weights."""
    spec = resolve_pretrained_spec(cfg)
    if spec is None:
        return None
    dataset = resolve_dataset_config(cfg)
    for key in ("mean", "std"):
        actual = dataset[f"image_{key}"]
        expected = spec[key]
        if len(actual) != len(expected) or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=atol)
            for a, b in zip(actual, expected)
        ):
            raise ValueError(
                f"Pretrained normalization mismatch for image_{key}: config has {actual}, "
                f"but {spec['source']} requires {expected}"
            )
    return spec


def build_model_from_config(
    cfg: Dict[str, Any],
    *,
    load_pretrained: bool | None = None,
) -> HPCLite:
    """Construct HPCLite; optionally suppress weight I/O when a task checkpoint will be loaded.

    ``model.pretrained`` remains part of checkpoint provenance and compatibility.  The
    ``load_pretrained`` override controls only whether timm downloads/loads those weights
    for this construction call.
    """
    model_cfg = cfg.get("model", cfg)
    unknown_keys = set(model_cfg.keys()) - _KNOWN_MODEL_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown keys in model config: {sorted(unknown_keys)}")

    resolved = resolve_model_config(cfg)
    requested_pretrained = resolved["pretrained"]
    effective_pretrained = (
        requested_pretrained if load_pretrained is None else bool(load_pretrained)
    )
    model = HPCLite(
        backbone_name=resolved["backbone"],
        pretrained=effective_pretrained,
        neck_width=resolved["neck_width"],
        context_dilations=resolved["context_dilations"],
        use_p8_context=resolved["use_p8_context"],
        use_repblock=resolved["use_repblock"],
        eps_d=resolved["eps_d"],
        output_stride=resolved["output_stride"],
        feature_reductions=tuple(int(name[1:]) for name in resolved["features"]),
    )
    model.pretrained_requested = requested_pretrained
    model.pretrained_loaded = effective_pretrained
    return model


def assert_checkpoint_compatible(checkpoint: dict, cfg: dict) -> None:
    """Assert that the checkpoint's embedded configuration matches the active YAML config."""
    trained = checkpoint.get("config")
    if trained is None or not isinstance(trained, dict):
        return

    old_model = resolve_model_config(trained)
    new_model = resolve_model_config(cfg)
    for k in old_model:
        if old_model[k] != new_model[k]:
            raise ValueError(
                f"Model config mismatch for '{k}': checkpoint has {old_model[k]!r}, active config has {new_model[k]!r}"
            )

    old_ds = resolve_dataset_config(trained)
    new_ds = resolve_dataset_config(cfg)
    if old_ds["name"] and new_ds["name"]:
        for k in old_ds:
            if old_ds[k] != new_ds[k]:
                raise ValueError(
                    f"Dataset config mismatch for '{k}': checkpoint has {old_ds[k]!r}, active config has {new_ds[k]!r}"
                )
