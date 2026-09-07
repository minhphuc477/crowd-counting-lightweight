from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import yaml


def _create_synthetic_dataset(root: Path) -> tuple[Path, Path]:
    images_dir = root / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = root / 'train.jsonl'
    val_manifest = root / 'val.jsonl'

    train_rows = []
    for i in range(2):
        img_path = images_dir / f'train_{i}.jpg'
        img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
        img.save(img_path)
        pts = [[20.0 + 10.0 * i, 30.0 + 10.0 * i], [50.0, 60.0]]
        train_rows.append({'image': str(img_path.resolve()), 'points': pts, 'id': f'train_{i}'})

    with train_manifest.open('w', encoding='utf-8') as f:
        for r in train_rows:
            f.write(json.dumps(r) + '\n')

    val_rows = []
    val_img_path = images_dir / 'val_0.jpg'
    val_img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    val_img.save(val_img_path)
    val_pts = [[40.0, 45.0], [70.0, 80.0]]
    val_rows.append({'image': str(val_img_path.resolve()), 'points': val_pts, 'id': 'val_0'})

    with val_manifest.open('w', encoding='utf-8') as f:
        for r in val_rows:
            f.write(json.dumps(r) + '\n')

    return train_manifest, val_manifest


def test_rmr_v3_cli_train_and_eval(tmp_path: Path) -> None:
    train_manifest, val_manifest = _create_synthetic_dataset(tmp_path)
    output_dir = tmp_path / 'run_smoke'

    cfg = {
        'model': {
            'output_stride': 4,
            'feature_width': 32,
            'pretrained': False,
            'region_sizes_px': [32, 64, 128],
            'iterations': 1,
            'omega': 1.0,
            'residual_clip': 0.0,
            'dispersion_init': 50.0,
            'dispersion_min': 0.5,
            'dispersion_max': 500.0,
            'reliability_mode': 'nb_rate_variance',
            'reliability_weight_min': 0.25,
            'reliability_weight_max': 4.0,
            'reliability_rate_std_floor': 0.01,
            'init_m0': 0.015763,
            'include_full_image': False,
        },
        'data': {
            'train_manifest': str(train_manifest),
            'val_manifest': str(val_manifest),
            'crop_size': 64,
            'scale_range': [0.9, 1.1],
        },
        'train': {
            'epochs': 2,
            'batch_size': 2,
            'workers': 0,
            'lr': 1e-4,
            'eval_every': 1,
            'solver_warmup_epochs': 0,
            'solver_ramp_epochs': 1,
            'amp': False,
            'deterministic': True,
            'early_stopping': False,
        },
        'output_dir': str(output_dir),
    }

    cfg_file = tmp_path / 'smoke_cfg.yaml'
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 1. Run train CLI
    cmd_train = [
        sys.executable,
        '-m',
        'rmr_v3.train',
        '--config',
        str(cfg_file),
        '--deterministic',
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    assert res_train.returncode == 0, f'Training CLI failed with stderr:\n{res_train.stderr}\nstdout:\n{res_train.stdout}'

    # Verify artifacts produced
    assert (output_dir / 'best_val_mae.pt').exists(), 'best_val_mae.pt was not saved!'
    assert (output_dir / 'last.pt').exists(), 'last.pt was not saved!'
    assert (output_dir / 'eval_val' / 'summary.json').exists(), 'eval_val/summary.json was not saved!'
    assert (output_dir / 'train_log.csv').exists(), 'train_log.csv was not saved!'

    summary = json.loads((output_dir / 'eval_val' / 'summary.json').read_text())
    assert 'MAE' in summary or 'mae' in summary

    # 2. Run eval CLI
    eval_out = output_dir / 'eval_standalone'
    cmd_eval = [
        sys.executable,
        '-m',
        'rmr_v3.eval',
        '--checkpoint',
        str(output_dir / 'best_val_mae.pt'),
        '--manifest',
        str(val_manifest),
        '--output-dir',
        str(eval_out),
        '--no-tiling',
    ]
    res_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
    assert res_eval.returncode == 0, f'Eval CLI failed with stderr:\n{res_eval.stderr}\nstdout:\n{res_eval.stdout}'

    # Predictions and summary should exist in eval_out
    assert (eval_out / 'predictions.csv').exists(), f'predictions.csv was not generated in {eval_out}!'
    assert (eval_out / 'summary.json').exists(), f'summary.json was not generated in {eval_out}!'


def test_rmr_v3_cli_profile(tmp_path: Path) -> None:
    """Verify rmr_v3.profile runs via CLI and outputs structured profiling results."""
    out_json = tmp_path / "profile_out.json"
    cmd = [
        sys.executable,
        "-m",
        "rmr_v3.profile",
        "--iterations",
        "2",
        "--height",
        "128",
        "--width",
        "128",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--device",
        "cpu",
        "--output",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Profile CLI failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["architecture"] == "RMR-v3"
    assert data["trainable_parameters"] == 101763
    assert data["input_shape"] == [1, 3, 128, 128]
    assert "latency_ms_mean" in data["fp32"]


def test_rmr_v3_cli_profile_override(tmp_path: Path) -> None:
    """Verify --weighted-reliability and --uniform-reliability correctly override config defaults."""
    cfg_uniform = tmp_path / "uniform.yaml"
    cfg_uniform.write_text(yaml.safe_dump({
        "model": {
            "iterations": 1,
            "region_sizes_px": [32, 64, 128],
            "uniform_reliability": True,
            "pretrained": False,
        }
    }))

    out_json = tmp_path / "prof_override.json"
    cmd = [
        sys.executable, "-m", "rmr_v3.profile",
        "--config", str(cfg_uniform),
        "--weighted-reliability",
        "--height", "64", "--width", "64",
        "--warmup", "1", "--iters", "1",
        "--device", "cpu",
        "--output", str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Profile failed: {res.stderr}"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["reliability_mode"] == "weighted", f"Expected weighted override, got {data['reliability_mode']}"

    # Now verify --uniform-reliability overrides a weighted config
    cfg_weighted = tmp_path / "weighted.yaml"
    cfg_weighted.write_text(yaml.safe_dump({
        "model": {
            "iterations": 1,
            "region_sizes_px": [32, 64, 128],
            "uniform_reliability": False,
            "pretrained": False,
        }
    }))
    cmd2 = [
        sys.executable, "-m", "rmr_v3.profile",
        "--config", str(cfg_weighted),
        "--uniform-reliability",
        "--height", "64", "--width", "64",
        "--warmup", "1", "--iters", "1",
        "--device", "cpu",
        "--output", str(out_json),
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0, f"Profile failed: {res2.stderr}"
    data2 = json.loads(out_json.read_text(encoding="utf-8"))
    assert data2["reliability_mode"] == "uniform", f"Expected uniform override, got {data2['reliability_mode']}"

