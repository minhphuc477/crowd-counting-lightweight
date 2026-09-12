"""DeepCoder Audit & Regression Test Suite: RMR-v9 Configs, ONNX Export, and Profiling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import yaml

from rmr_v3.config import load_config, validate_v3_config
from rmr_v3.profile import count_trainable_parameters, measure_clean_peak_memory, profile_latency, profiler_flops
from rmr_v3.train import make_loss_cfg, make_model
from tools.export_onnx import _ExportWrapper, build_model_from_config, export_model_to_onnx, is_rmr_config


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "rmr_v9"

EXPECTED_CONFIG_NAMES = [
    "rmr_v9_canonical.yaml",
    "rmr_v9_aq_rmr.yaml",
    "rmr_v9_ablation_no_proximal.yaml",
    "rmr_v9_ablation_isotropic.yaml",
    "rmr_v9_ablation_mean_only.yaml",
    "rmr_v9_control_no_solver.yaml",
]


@pytest.mark.parametrize("config_name", EXPECTED_CONFIG_NAMES)
def test_rmr_v9_config_exact_spec_and_parameter_bounds(config_name: str):
    """Verify each RMR-v9 YAML config validates, instantiates, and respects <=105,000 params."""
    cfg_path = CONFIG_DIR / config_name
    assert cfg_path.is_file(), f"Config file missing: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Configuration validation
    validate_v3_config(cfg)

    # 2. Architecture instantiation and parameter bounds
    model, uniform_reliability = make_model(cfg)
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())

    assert trainable_params <= 105_000, (
        f"{config_name} exceeds parameter budget: {trainable_params} > 105,000"
    )
    assert trainable_params == total_params, (
        f"{config_name} has frozen/non-trainable parameters: {trainable_params} != {total_params}"
    )

    # 3. Training hyperparameter validity
    train_cfg = cfg.get("train", {})
    assert float(train_cfg.get("lr", 0.0)) > 0.0, f"Invalid lr in {config_name}"
    assert int(train_cfg.get("epochs", 0)) > 0, f"Invalid epochs in {config_name}"
    assert int(train_cfg.get("batch_size", 0)) > 0, f"Invalid batch_size in {config_name}"
    assert float(train_cfg.get("backbone_lr_scale", 0.0)) > 0.0, f"Invalid backbone_lr_scale in {config_name}"

    # 4. Loss schedule validity
    loss_cfg = make_loss_cfg(cfg)
    assert loss_cfg.allocation_loss_type == "flat_dm16"
    assert loss_cfg.count_loss_mode == "nb"
    assert loss_cfg.lambda_count > 0.0
    assert loss_cfg.lambda_flat_dm16 > 0.0

    # 5. Data manifest path resolution
    data_cfg = cfg.get("data", {})
    train_man = data_cfg.get("train_manifest")
    val_man = data_cfg.get("val_manifest")
    assert train_man and (REPO_ROOT / train_man).is_file(), f"Missing train manifest: {train_man}"
    assert val_man and (REPO_ROOT / val_man).is_file(), f"Missing val manifest: {val_man}"


def test_is_rmr_config_detection():
    """Verify is_rmr_config correctly detects RMR vs MICF configurations."""
    rmr_cfg = {"model": {"region_sizes_px": [32, 64], "neck_type": "additive"}}
    micf_cfg = {"model": {"backbone": "mobilenetv4", "neck_width": 32}}
    assert is_rmr_config(rmr_cfg) is True
    assert is_rmr_config(micf_cfg) is False


def test_rmr_v9_profiling_utilities():
    """Test rmr_v3/profile.py functions on rmr_v9_canonical."""
    cfg = load_config(str(CONFIG_DIR / "rmr_v9_canonical.yaml"))
    model, _ = make_model(cfg)
    model.eval()

    x = torch.randn(1, 3, 256, 256)

    # Latency profiling
    lat = profile_latency(model, x, warmup=2, iters=3, use_amp=False)
    assert "latency_ms_mean" in lat
    assert lat["latency_ms_mean"] > 0.0
    assert lat["fps_from_mean"] > 0.0

    # FLOPs profiling
    flops = profiler_flops(model, x)
    assert flops is not None
    assert flops > 0.0


def test_rmr_v9_onnx_export_and_runtime_parity():
    """Verify RMRv3 model exports to ONNX in deploy mode with <1e-4 parity against PyTorch."""
    ort = pytest.importorskip("onnxruntime")

    cfg_path = str(CONFIG_DIR / "rmr_v9_canonical.yaml")
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = str(Path(tmpdir) / "test_rmr.onnx")

        export_model_to_onnx(
            checkpoint_path=None,
            config_path=cfg_path,
            output_onnx=onnx_file,
            input_resolution=256,
            opset_version=17,
            allow_random_init=True,
            deploy_mode=True,
            skip_verify=False,
        )

        assert Path(onnx_file).is_file()
        assert Path(onnx_file).stat().st_size > 100_000


def test_summarize_rmr_v9_suite_script():
    """Verify summarize_rmr_v9_suite parses directories and generates markdown & CSV."""
    from scripts.summarize_rmr_v9_suite import main as summarize_main
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_runs = Path(tmpdir) / "runs"
        run_dir = tmp_runs / "rmr_v9_test_run"
        run_dir.mkdir(parents=True)

        # Create mock train_log.csv
        csv_file = run_dir / "train_log.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "val_mae"])
            w.writerow(["1", "75.5"])
            w.writerow(["2", "68.2"])

        # Create mock summary.json
        summary_file = run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "mae": 68.2,
                "rmse": 105.4,
                "nae": 0.25,
                "bias": -1.2,
                "game0": 68.2,
                "game1": 85.0,
                "game2": 95.0,
                "game3": 110.0,
                "best_epoch": 2,
            }, f)

        out_md = str(Path(tmpdir) / "summary.md")
        out_csv = str(Path(tmpdir) / "summary.csv")

        orig_argv = sys.argv
        try:
            sys.argv = [
                "summarize_rmr_v9_suite.py",
                "--runs-dir", str(tmp_runs),
                "--pattern", "rmr_v9",
                "--output-md", out_md,
                "--output-csv", out_csv,
            ]
            summarize_main()
        finally:
            sys.argv = orig_argv

        assert Path(out_md).is_file()
        assert Path(out_csv).is_file()
        md_text = Path(out_md).read_text(encoding="utf-8")
        assert "rmr_v9_test_run" in md_text
        assert "68.2" in md_text


def test_rmr_v9_aq_rmr_onnx_export_parity():
    """Verify rmr_v9_aq_rmr (anisotropic boxes, mean_std stats, tau=0.015) exports to ONNX cleanly."""
    pytest.importorskip("onnxruntime")

    cfg_path = str(CONFIG_DIR / "rmr_v9_aq_rmr.yaml")
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = str(Path(tmpdir) / "test_aq_rmr.onnx")

        export_model_to_onnx(
            checkpoint_path=None,
            config_path=cfg_path,
            output_onnx=onnx_file,
            input_resolution=256,
            opset_version=17,
            allow_random_init=True,
            deploy_mode=True,
            skip_verify=False,
        )

        assert Path(onnx_file).is_file()
        assert Path(onnx_file).stat().st_size > 100_000


def test_summarize_rmr_v9_suite_csv_fallback():
    """Verify summarizer falls back to train_log.csv best validation metrics when summary.json is absent."""
    from scripts.summarize_rmr_v9_suite import main as summarize_main
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_runs = Path(tmpdir) / "runs"
        run_dir = tmp_runs / "rmr_v9_fallback_run"
        run_dir.mkdir(parents=True)

        # Create train_log.csv WITHOUT summary.json
        csv_file = run_dir / "train_log.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "val_mae", "val_rmse", "val_nae", "val_bias"])
            w.writerow(["1", "80.0", "120.0", "0.30", "-2.0"])
            w.writerow(["2", "64.5", "101.2", "0.22", "-0.8"])

        out_md = str(Path(tmpdir) / "summary.md")
        out_csv = str(Path(tmpdir) / "summary.csv")

        orig_argv = sys.argv
        try:
            sys.argv = [
                "summarize_rmr_v9_suite.py",
                "--runs-dir", str(tmp_runs),
                "--pattern", "rmr_v9",
                "--output-md", out_md,
                "--output-csv", out_csv,
            ]
            summarize_main()
        finally:
            sys.argv = orig_argv

        md_text = Path(out_md).read_text(encoding="utf-8")
        assert "rmr_v9_fallback_run" in md_text
        assert "64.50" in md_text or "64.5" in md_text


def test_bash_matrix_script_structure_and_configs():
    """Verify run_rmr_v9_matrix_ubuntu.sh references all 6 configs and redirects diagnostic echos to stderr."""
    script_path = REPO_ROOT / "scripts" / "run_rmr_v9_matrix_ubuntu.sh"
    assert script_path.is_file()

    content = script_path.read_text(encoding="utf-8")

    # Check all 6 runs are registered
    for name in EXPECTED_CONFIG_NAMES:
        stem = Path(name).stem
        assert stem in content, f"Missing {stem} in run_rmr_v9_matrix_ubuntu.sh"

    # Verify diagnostic messages in launch_single_run are redirected to stderr
    assert 'echo "  [$run_id] Logging to: $log_file" >&2' in content
    assert 'echo "$pid"' in content


def test_powershell_matrix_script_structure_and_configs():
    """Verify run_rmr_v9_matrix.ps1 references all 6 configs and handles resume and auto-evaluation."""
    script_path = REPO_ROOT / "scripts" / "run_rmr_v9_matrix.ps1"
    assert script_path.is_file()

    content = script_path.read_text(encoding="utf-8")
    for name in EXPECTED_CONFIG_NAMES:
        stem = Path(name).stem
        assert stem in content, f"Missing {stem} in run_rmr_v9_matrix.ps1"

    assert "rmr_v3.eval" in content
    assert "summarize_rmr_v9_suite.py" in content


def test_architecture_table_rmr_support():
    """Verify tools/architecture_table.py runs cleanly on RMR models and respects parameter budget."""
    from tools.architecture_table import collect_architecture, write_markdown

    with open(CONFIG_DIR / "rmr_v9_canonical.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = collect_architecture(cfg, height=128, width=128)
    assert result["total_params"] == 101_763
    assert result["is_rmr"] is True
    assert "backbone" in result["component_params"]
    assert "fine_head" in result["component_params"]
    assert "neck" in result["component_params"]
    assert "region_head" in result["component_params"]
    assert result["total_macs"] > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_md = Path(tmpdir) / "arch.md"
        write_markdown(str(out_md), str(CONFIG_DIR / "rmr_v9_canonical.yaml"), result, height=128, width=128)
        content = out_md.read_text(encoding="utf-8")
        assert "RMR Executed Architecture Table" in content
        assert "101,763" in content


def test_profile_model_rmr_support():
    """Verify tools/profile_model.py runs efficiency profiling on RMR models."""
    from tools.profile_model import build_model_from_config, profile_model_efficiency

    with open(CONFIG_DIR / "rmr_v9_canonical.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    profile = profile_model_efficiency(model, input_resolution=128, batch_size=1, warmup_iters=1, measure_iters=2)
    assert profile["params_total"] == 101_763
    assert profile["params_trainable"] == 101_763
    assert profile["conv_macs"] > 0
    assert profile["latency_median_ms"] > 0.0


