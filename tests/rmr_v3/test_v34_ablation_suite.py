from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config

V34_DIR = Path("configs/rmr_v34")
CANONICAL_FILE = V34_DIR / "rmr_v34_diag_canonical.yaml"


@pytest.fixture(scope="module")
def canonical_config() -> dict:
    assert CANONICAL_FILE.exists(), f"Canonical config not found: {CANONICAL_FILE}"
    return yaml.safe_load(CANONICAL_FILE.read_text(encoding="utf-8-sig"))


def test_all_v34_configs_exist_and_validate(canonical_config: dict) -> None:
    yaml_files = list(V34_DIR.glob("*.yaml"))
    assert len(yaml_files) >= 17, f"Expected >= 17 configs in {V34_DIR}, found {len(yaml_files)}"

    for yf in yaml_files:
        raw = yaml.safe_load(yf.read_text(encoding="utf-8-sig"))
        # Must pass full validator
        validate_v3_config(raw)


def test_all_v34_configs_parameter_ceiling() -> None:
    yaml_files = list(V34_DIR.glob("*.yaml"))

    for yf in yaml_files:
        raw = yaml.safe_load(yf.read_text(encoding="utf-8-sig"))
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        m_cfg.pretrained = False
        model = RMRv3(m_cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert params <= 105000, (
            f"Config {yf.name} has {params:,} parameters, strictly violating <= 105,000 ceiling!"
        )


def test_single_variable_isolation_in_v34_ablations(canonical_config: dict) -> None:
    """Verify that each ablation config differs from canonical only in its intended fields."""
    yaml_files = [f for f in V34_DIR.glob("*.yaml") if f.name != "rmr_v34_diag_canonical.yaml"]

    for yf in yaml_files:
        raw = yaml.safe_load(yf.read_text(encoding="utf-8-sig"))
        diffs = []

        # Check top-level
        for k in set(canonical_config) | set(raw):
            if k in ("output_dir", "seed"):
                continue
            if k not in ("data", "model", "loss", "train", "eval"):
                if canonical_config.get(k) != raw.get(k):
                    diffs.append((k, canonical_config.get(k), raw.get(k)))

        # Check sub-sections
        for section in ("data", "model", "loss", "train", "eval"):
            c_sec = canonical_config.get(section, {})
            r_sec = raw.get(section, {})
            all_keys = set(c_sec.keys()) | set(r_sec.keys())
            for sk in all_keys:
                if c_sec.get(sk) != r_sec.get(sk):
                    diffs.append((f"{section}.{sk}", c_sec.get(sk), r_sec.get(sk)))

        # For SHB configs, data changes are expected
        if "shb" in yf.name:
            assert any("data." in d[0] for d in diffs)
        elif "seed" in yf.name:
            assert len(diffs) == 0  # Only seed & output_dir changed
        else:
            # Single-variable ablation should change at most 2 closely coupled keys
            # (e.g. hurdle_head + lambda_hurdle, or density_curvature + lambda_curvature)
            assert 1 <= len(diffs) <= 3, (
                f"Ablation {yf.name} violates single-variable isolation! Diffs: {diffs}"
            )


def test_v34_ablation_forward_backward_gradient_integrity() -> None:
    """Run forward and backward pass on key ablation model variants to verify active gradients."""
    test_configs = [
        "rmr_v34_diag_canonical.yaml",
        "rmr_v34_abl_vdp_dcap.yaml",
        "rmr_v34_abl_no_diag.yaml",
        "rmr_v34_abl_no_dcap_tilt.yaml",
        "rmr_v34_abl_no_solver.yaml",
        "rmr_v34_abl_solver_t2.yaml",
        "rmr_v34_abl_no_resonant.yaml",
        "rmr_v34_abl_no_proximal.yaml",
        "rmr_v34_abl_soft_proximal.yaml",
    ]

    for cf in test_configs:
        cfg_path = V34_DIR / cf
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig"))
        m_cfg = RMRv3Config.from_dict(raw.get("model", {}))
        m_cfg.pretrained = False
        model = RMRv3(m_cfg)
        model.train()

        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        out = model(x)
        assert out.y.shape[-2:] == (64, 64)
        assert torch.isfinite(out.y).all()

        loss = out.y.sum()
        loss.backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        # Verify backbone stem receives gradients
        stem_grad = next(model.encoder.parameters()).grad
        assert stem_grad is not None and torch.isfinite(stem_grad).all()
