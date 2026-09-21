"""Tests for RMR-v30 Milestone 2: Configs, Bitwise v19 Parity & Monolith Prevention.

Verifies:
1. Configuration loading, schema validity, and zero KD across the 6-model ablation suite.
2. Single-variable ladder isolation across all 6 model configurations.
3. Trainable parameter ceiling (<= 105,000) on all 6 models.
4. Exact bitwise parity (diff == 0.000000) between v19 canonical isotropic and v30 step0 anchor.
5. Strict monolith prevention (<= 450 lines) across all source files in rmr_core/ and rmr_v3/.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.config import load_config
from rmr_v3.losses import compute_rmr_v3_losses, RMRv3LossConfig
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.dual_lattice import check_mass_conservation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "rmr_v30"

CONFIG_NAMES = [
    "rmr_v30_step0_v19_anchor",
    "rmr_v30_h1_anscombe_sirt",
    "rmr_v30_h2_dual_lattice_dcsr",
    "rmr_v30_h3_anscombe_dual_lattice",
    "rmr_v30_h4_deep_sirt_t8",
    "rmr_v30_control_no_solver",
]


def _diff_dicts(d1: dict, d2: dict, path: str = "") -> list[tuple[str, object, object]]:
    diffs = []
    all_keys = set(d1.keys()).union(d2.keys())
    for k in sorted(all_keys):
        subpath = f"{path}.{k}" if path else k
        if k not in d1:
            diffs.append((subpath, None, d2[k]))
        elif k not in d2:
            diffs.append((subpath, d1[k], None))
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            diffs.extend(_diff_dicts(d1[k], d2[k], subpath))
        elif d1[k] != d2[k]:
            diffs.append((subpath, d1[k], d2[k]))
    return diffs


class TestRMRv30Configs:
    """Validate existence, syntax, schema, and Zero KD of all 6 ablation configs."""

    @pytest.mark.parametrize("cfg_name", CONFIG_NAMES)
    def test_config_loads_and_validates(self, cfg_name: str):
        cfg_path = CONFIG_DIR / f"{cfg_name}.yaml"
        assert cfg_path.exists(), f"Missing config file: {cfg_path}"
        cfg = load_config(cfg_path)
        assert isinstance(cfg, dict)
        assert cfg["seed"] == 42
        assert "model" in cfg and "loss" in cfg and "train" in cfg and "data" in cfg

    @pytest.mark.parametrize("cfg_name", CONFIG_NAMES)
    def test_zero_knowledge_distillation_contract(self, cfg_name: str):
        """Zero KD: strictly 0 teacher checkpoints loaded in Stages 1 and 2."""
        cfg_path = CONFIG_DIR / f"{cfg_name}.yaml"
        cfg = load_config(cfg_path)
        train_cfg = cfg.get("train", {})
        assert "teacher_ckpt" not in train_cfg or not train_cfg["teacher_ckpt"], (
            f"Zero KD violation in {cfg_name}: teacher_ckpt is declared!"
        )

    @pytest.mark.parametrize("cfg_name", CONFIG_NAMES)
    def test_canonical_dataset_manifest_contract(self, cfg_name: str):
        """All models must train on canonical ShanghaiTech Part A partition."""
        cfg_path = CONFIG_DIR / f"{cfg_name}.yaml"
        cfg = load_config(cfg_path)
        data_cfg = cfg["data"]
        assert data_cfg["train_manifest"] == "data/sha_a_train_all.jsonl"
        assert data_cfg["val_manifest"] == "data/sha_a_test.jsonl"


class TestSingleVariableIsolation:
    """Verify clean two-loop ablation ladder with zero cross-talk."""

    @pytest.fixture
    def loaded_configs(self) -> dict[str, dict]:
        return {name: yaml.safe_load(open(CONFIG_DIR / f"{name}.yaml")) for name in CONFIG_NAMES}

    def test_step0_to_h1_anscombe_sirt_isolation(self, loaded_configs: dict[str, dict]):
        """Step 0 -> H1 varies ONLY Anscombe VST parameters on Stride 4."""
        diffs = [
            x for x in _diff_dicts(loaded_configs["rmr_v30_step0_v19_anchor"], loaded_configs["rmr_v30_h1_anscombe_sirt"])
            if x[0] != "output_dir"
        ]
        diff_keys = {x[0] for x in diffs}
        expected_keys = {
            "model.adjoint_mode",
            "model.use_anscombe_sirt",
            "model.anscombe_c",
        }
        assert diff_keys == expected_keys, f"Unexpected cross-talk in H1: {diff_keys ^ expected_keys}"
        assert loaded_configs["rmr_v30_h1_anscombe_sirt"]["model"]["adjoint_mode"] == "anscombe_vst"
        assert loaded_configs["rmr_v30_h1_anscombe_sirt"]["model"]["use_anscombe_sirt"] is True

    def test_step0_to_h2_dual_lattice_isolation(self, loaded_configs: dict[str, dict]):
        """Step 0 -> H2 varies ONLY Dual-Lattice DCSR parameters."""
        diffs = [
            x for x in _diff_dicts(loaded_configs["rmr_v30_step0_v19_anchor"], loaded_configs["rmr_v30_h2_dual_lattice_dcsr"])
            if x[0] != "output_dir"
        ]
        diff_keys = {x[0] for x in diffs}
        expected_keys = {
            "model.output_stride",
            "model.subpixel_stride2",
            "model.adaptive_tau",
            "model.adaptive_tau_rho0",
            "loss.lambda_carrier_cell",  # intrinsic to subpixel_stride2 mode
            "loss.lambda_fine_cell",     # intrinsic to subpixel_stride2 mode
        }
        assert diff_keys == expected_keys, f"Unexpected cross-talk in H2: {diff_keys ^ expected_keys}"
        assert loaded_configs["rmr_v30_h2_dual_lattice_dcsr"]["model"]["output_stride"] == 2
        assert loaded_configs["rmr_v30_h2_dual_lattice_dcsr"]["model"]["subpixel_stride2"] is True
        assert loaded_configs["rmr_v30_h2_dual_lattice_dcsr"]["model"]["adjoint_mode"] == "radon_nikodym"

    def test_h1_plus_h2_composite_h3_verification(self, loaded_configs: dict[str, dict]):
        """H3 is the exact union of H1 (Anscombe VST) + H2 (Dual-Lattice DCSR)."""
        diffs = [
            x for x in _diff_dicts(loaded_configs["rmr_v30_step0_v19_anchor"], loaded_configs["rmr_v30_h3_anscombe_dual_lattice"])
            if x[0] != "output_dir"
        ]
        diff_keys = {x[0] for x in diffs}
        expected_keys = {
            "model.adjoint_mode",
            "model.use_anscombe_sirt",
            "model.anscombe_c",
            "model.output_stride",
            "model.subpixel_stride2",
            "model.adaptive_tau",
            "model.adaptive_tau_rho0",
            "loss.lambda_carrier_cell",  # intrinsic to subpixel_stride2 mode
            "loss.lambda_fine_cell",     # intrinsic to subpixel_stride2 mode
        }
        assert diff_keys == expected_keys, f"H3 composite mismatch: {diff_keys ^ expected_keys}"

    def test_h3_to_h4_deep_sirt_isolation(self, loaded_configs: dict[str, dict]):
        """H3 -> H4 varies ONLY iterations: 6 -> 8."""
        diffs = [
            x for x in _diff_dicts(loaded_configs["rmr_v30_h3_anscombe_dual_lattice"], loaded_configs["rmr_v30_h4_deep_sirt_t8"])
            if x[0] != "output_dir"
        ]
        diff_keys = {x[0] for x in diffs}
        assert diff_keys == {"model.iterations"}, f"Unexpected cross-talk in H4: {diff_keys}"
        assert loaded_configs["rmr_v30_h4_deep_sirt_t8"]["model"]["iterations"] == 8

    def test_step0_to_control_no_solver_isolation(self, loaded_configs: dict[str, dict]):
        """Step 0 -> Control varies ONLY enable_solver: True -> False."""
        diffs = [
            x for x in _diff_dicts(loaded_configs["rmr_v30_step0_v19_anchor"], loaded_configs["rmr_v30_control_no_solver"])
            if x[0] != "output_dir"
        ]
        diff_keys = {x[0] for x in diffs}
        assert diff_keys == {"model.enable_solver"}, f"Unexpected cross-talk in Control: {diff_keys}"
        assert loaded_configs["rmr_v30_control_no_solver"]["model"]["enable_solver"] is False


class TestTrainableParameterCeiling:
    """Verify all 6 models strictly obey <= 105,000 trainable parameters."""

    @pytest.mark.parametrize(
        "cfg_name,expected_params",
        [
            ("rmr_v30_step0_v19_anchor", 104441),
            ("rmr_v30_h1_anscombe_sirt", 104441),
            ("rmr_v30_h2_dual_lattice_dcsr", 104540),
            ("rmr_v30_h3_anscombe_dual_lattice", 104540),
            ("rmr_v30_h4_deep_sirt_t8", 104540),
            ("rmr_v30_control_no_solver", 104441),
        ],
    )
    def test_parameter_count_exactness_and_ceiling(self, cfg_name: str, expected_params: int):
        cfg = load_config(CONFIG_DIR / f"{cfg_name}.yaml")
        model = RMRv3(RMRv3Config.from_dict(cfg["model"]))
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params <= 105000, f"Budget exceeded in {cfg_name}: {n_params} > 105,000"
        assert n_params == expected_params, f"Unexpected parameter count in {cfg_name}: {n_params} != {expected_params}"


class TestBitwiseBaselineParity:
    """Verify exact bitwise parity (diff == 0.000000) between v19 canonical and v30 step0."""

    @pytest.fixture(scope="class")
    def paired_models(self):
        c19 = yaml.safe_load(open(REPO_ROOT / "configs" / "rmr_v19" / "rmr_v19_canonical_isotropic.yaml"))
        c30 = yaml.safe_load(open(CONFIG_DIR / "rmr_v30_step0_v19_anchor.yaml"))
        torch.manual_seed(42)
        m19 = RMRv3(RMRv3Config.from_dict(c19["model"])).eval()
        torch.manual_seed(42)
        m30 = RMRv3(RMRv3Config.from_dict(c30["model"])).eval()
        return m19, m30, c19["loss"], c30["loss"]

    def test_forward_output_bitwise_parity(self, paired_models):
        m19, m30, _, _ = paired_models
        for shape in [(1, 3, 256, 256), (2, 3, 384, 384)]:
            torch.manual_seed(777)
            x = torch.randn(*shape)
            with torch.no_grad():
                torch.manual_seed(888)
                o19 = m19(x)
                torch.manual_seed(888)
                o30 = m30(x)

            diff_y = (o19.y - o30.y).abs().max().item()
            diff_y0 = (o19.y0 - o30.y0).abs().max().item()
            assert diff_y == 0.0, f"Forward y difference on {shape}: {diff_y}"
            assert diff_y0 == 0.0, f"Forward y0 difference on {shape}: {diff_y0}"

    def test_loss_components_bitwise_parity(self, paired_models):
        m19, m30, lcfg19, lcfg30 = paired_models
        torch.manual_seed(777)
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            torch.manual_seed(888)
            o19 = m19(x)
            torch.manual_seed(888)
            o30 = m30(x)

        tgt = torch.zeros(2, 1, 64, 64)
        tgt[0, 0, 15, 15] = 1.0
        tgt[1, 0, 30, 40] = 3.5

        loss19 = compute_rmr_v3_losses(o19, tgt, RMRv3LossConfig.from_dict(lcfg19))
        loss30 = compute_rmr_v3_losses(o30, tgt, RMRv3LossConfig.from_dict(lcfg30))

        assert len(loss19) >= 18, f"Expected >= 18 loss components, got {len(loss19)}"
        assert set(loss19.keys()) == set(loss30.keys())

        for k in sorted(loss19.keys()):
            diff_k = (loss19[k] - loss30[k]).abs().item()
            assert diff_k == 0.0, f"Loss parity mismatch on {k}: diff = {diff_k}"


class TestCodebaseMonolithPrevention:
    """Verify all Python source files in rmr_core/ and rmr_v3/ are strictly <= 450 lines."""

    def test_source_file_line_counts(self):
        source_dirs = [REPO_ROOT / "rmr_core", REPO_ROOT / "rmr_v3"]
        audited_files = []
        violators = []

        for sdir in source_dirs:
            for root, _, files in os.walk(sdir):
                if "__pycache__" in root:
                    continue
                for fname in files:
                    if fname.endswith(".py"):
                        fpath = Path(root) / fname
                        line_count = len(fpath.read_text(encoding="utf-8").splitlines())
                        audited_files.append((str(fpath.relative_to(REPO_ROOT)), line_count))
                        if line_count > 450:
                            violators.append((str(fpath.relative_to(REPO_ROOT)), line_count))

        assert len(audited_files) == 58, f"Expected exactly 58 source files, found {len(audited_files)}"
        assert not violators, f"Monolith invariant violated! Files exceeding 450 lines: {violators}"


class TestDualLatticeInvariants:
    """Verify mathematical invariants for Dual-Lattice DCSR models."""

    def test_pushforward_mass_conservation(self):
        cfg = load_config(CONFIG_DIR / "rmr_v30_h3_anscombe_dual_lattice.yaml")
        model = RMRv3(RMRv3Config.from_dict(cfg["model"])).eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert "y_carrier" in out
        assert check_mass_conservation(out.y, out["y_carrier"], eps=1e-6)
