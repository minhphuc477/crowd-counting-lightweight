"""Automated Validation and Invariant Test Suite for RMR Research Experiment Suite (A*).

Validates:
1. Schema conformity via rmr_v3.config.validate_v3_config.
2. Exact parameter count constraint (104,441 trainable parameters).
3. Non-deterministic data entropy preservation (deterministic: false).
4. Strict Single-Variable Isolation against rmr_v19_canonical_isotropic baseline.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rmr_v3.config import validate_v3_config
from rmr_v3.engine import make_loss_cfg, make_model


RESEARCH_CONFIG_DIR = _REPO_ROOT / "configs" / "rmr_research"
CANONICAL_V19_PATH = _REPO_ROOT / "configs" / "rmr_v19" / "rmr_v19_canonical_isotropic.yaml"

CONFIG_FILES = [
    "h3a_depth2.yaml",
    "h3b_depth4.yaml",
    "h3c_depth8.yaml",
    "h4_no_morozov.yaml",
    "h5_no_curvature.yaml",
    "h6_spectral_composite.yaml",
    "h7_resonant_adjoint.yaml",
    "h8_harmonious_composite.yaml",
]


def load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"File does not exist: {path}"
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), f"YAML root must be a dict, got {type(data)}"
    return data


@pytest.fixture(scope="module")
def canonical_v19_cfg() -> dict[str, Any]:
    return load_yaml(CANONICAL_V19_PATH)


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_config_file_exists_and_validates(config_name: str):
    """Ensure config YAML exists and passes rmr_v3 schema validation."""
    cfg_path = RESEARCH_CONFIG_DIR / config_name
    cfg = load_yaml(cfg_path)
    # validate_v3_config raises ValueError if any key is unknown or invalid
    validate_v3_config(cfg)
    assert "output_dir" in cfg
    assert cfg["output_dir"].startswith("runs/sha_a/")


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_train_deterministic_is_false(config_name: str):
    """Ensure deterministic: false is explicitly set to preserve 300-image entropy."""
    cfg_path = RESEARCH_CONFIG_DIR / config_name
    cfg = load_yaml(cfg_path)
    assert "train" in cfg, f"'train' section missing in {config_name}"
    assert "deterministic" in cfg["train"], f"'deterministic' key missing in {config_name}"
    assert cfg["train"]["deterministic"] is False, (
        f"deterministic must be False in {config_name} to preserve data entropy!"
    )


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_exact_parameter_count_104441(config_name: str):
    """Ensure every experiment model instantiates with exactly 104,441 trainable parameters."""
    cfg_path = RESEARCH_CONFIG_DIR / config_name
    cfg = load_yaml(cfg_path)
    model, _ = make_model(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 104441, f"Expected exactly 104,441 parameters in {config_name}, got {n_params}"
    assert n_params <= 105000, f"Exceeded lightweight budget: {n_params} > 105,000"


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dictionary into dot-separated paths."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        else:
            out[key] = v
    return out


def test_single_variable_isolation(canonical_v19_cfg: dict[str, Any]):
    """Verify that each research config differs strictly in its designated isolated variables."""
    base_flat = _flatten_dict(canonical_v19_cfg)

    # Specific expected differences relative to canonical_v19_cfg:
    # Notice canonical_v19_cfg has no train.deterministic key (defaults to True in trainer),
    # while all research configs explicitly specify train.deterministic: false,
    # and output_dir points to their dedicated experiment run folder.
    expected_diffs = {
        "h3a_depth2.yaml": {
            "model.iterations": 2,
        },
        "h3b_depth4.yaml": {
            "model.iterations": 4,
        },
        "h3c_depth8.yaml": {
            "model.iterations": 8,
        },
        "h4_no_morozov.yaml": {
            "model.morozov_gamma": 0.0,
        },
        "h5_no_curvature.yaml": {
            "loss.lambda_curvature": 0.0,
        },
        "h6_spectral_composite.yaml": {
            "loss.use_spectral_loss": True,
            "loss.lambda_spectral": 0.2,
            "loss.spectral_beta": 2.0,
            "loss.lambda_spectral_dc": 1.0,
        },
        "h7_resonant_adjoint.yaml": {
            "model.resonant_adjoint": True,
            "model.resonant_adjoint_lambda": 0.5,
            "model.anscombe_morozov": True,
        },
        "h8_harmonious_composite.yaml": {
            "model.resonant_adjoint": True,
            "model.resonant_adjoint_lambda": 0.5,
            "model.anscombe_morozov": True,
            "loss.use_spectral_loss": True,
            "loss.lambda_spectral": 0.2,
            "loss.spectral_beta": 2.0,
            "loss.lambda_spectral_dc": 1.0,
            "loss.spectral_bandpass": True,
            "loss.spectral_omega_low": 0.02,
            "loss.spectral_omega_high": 0.35,
            "loss.regional_mass_weight_alpha": 1.5,
        },
    }

    for config_name, expected_vars in expected_diffs.items():
        cfg_path = RESEARCH_CONFIG_DIR / config_name
        cfg = load_yaml(cfg_path)
        flat = _flatten_dict(cfg)

        # Keys permitted to differ universally
        universal_allowed_diffs = {"output_dir", "train.deterministic", "train.workers"}

        all_keys = set(base_flat.keys()) | set(flat.keys())
        actual_diffs = {}
        for k in all_keys:
            if k in universal_allowed_diffs:
                continue
            val_base = base_flat.get(k)
            val_cfg = flat.get(k)
            if val_base != val_cfg:
                actual_diffs[k] = val_cfg

        assert actual_diffs == expected_vars, (
            f"Config {config_name} violates Single-Variable Isolation!\n"
            f"Expected only: {expected_vars}\n"
            f"Actual diffs: {actual_diffs}"
        )


def test_h6_spectral_loss_integration():
    """Verify H6 properly initializes CountPreservingSpectralLoss with correct hyperparameters."""
    cfg_path = RESEARCH_CONFIG_DIR / "h6_spectral_composite.yaml"
    cfg = load_yaml(cfg_path)
    loss_cfg = make_loss_cfg(cfg)

    assert loss_cfg.use_spectral_loss is True
    assert loss_cfg.lambda_spectral == 0.2
    assert loss_cfg.spectral_beta == 2.0
    assert loss_cfg.lambda_spectral_dc == 1.0


def test_codebase_line_count_invariant():
    """Verify test file itself adheres to strict protocol limit <= 450 lines."""
    this_file = Path(__file__).resolve()
    lines = this_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 450, f"{this_file.name} exceeds 450 lines: {len(lines)}"


if __name__ == "__main__":
    pytest.main([str(Path(__file__).resolve()), "-v"])
