"""Verification tests for NWPU-Crowd manifests and benchmark invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def test_nwpu_manifest_counts_and_splits(data_dir: Path):
    train_manifest = data_dir / "nwpu_train.jsonl"
    val_manifest = data_dir / "nwpu_val.jsonl"
    test_manifest = data_dir / "nwpu_test.jsonl"

    assert train_manifest.exists(), "nwpu_train.jsonl must exist"
    assert val_manifest.exists(), "nwpu_val.jsonl must exist"
    assert test_manifest.exists(), "nwpu_test.jsonl must exist"

    train_lines = train_manifest.read_text(encoding="utf-8").strip().splitlines()
    val_lines = val_manifest.read_text(encoding="utf-8").strip().splitlines()
    test_lines = test_manifest.read_text(encoding="utf-8").strip().splitlines()

    assert len(train_lines) == 3109, f"NWPU train must have 3109 samples, got {len(train_lines)}"
    assert len(val_lines) == 500, f"NWPU val must have 500 samples, got {len(val_lines)}"
    assert len(test_lines) == 1500, f"NWPU test must have 1500 samples, got {len(test_lines)}"


def test_nwpu_manifest_schema_and_negative_samples(data_dir: Path):
    train_manifest = data_dir / "nwpu_train.jsonl"
    val_manifest = data_dir / "nwpu_val.jsonl"

    zero_count_total = 0
    total_heads = 0

    for m_path in [train_manifest, val_manifest]:
        with open(m_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "id" in record
                assert "image" in record
                assert "points" in record
                pts = record["points"]
                cnt = record.get("count", len(pts))
                assert len(pts) == cnt
                total_heads += cnt
                if cnt == 0:
                    zero_count_total += 1
                elif len(pts) > 0:
                    for p in pts[:5]:
                        assert len(p) == 2
                        assert p[0] >= 0.0 and p[1] >= 0.0

    assert zero_count_total == 248, f"NWPU must contain exactly 248 zero-count samples, got {zero_count_total}"
    assert total_heads == 1488690, f"Total heads in NWPU (train+val) must be 1,488,690, got {total_heads}"


def test_nwpu_localization_benchmark_file(data_dir: Path):
    loc_file = data_dir / "nwpu_val_gt_loc.txt"
    assert loc_file.exists(), "Official val_gt_loc.txt benchmark file must exist"
    assert loc_file.stat().st_size > 3_000_000, "val_gt_loc.txt must be > 3MB"

    with open(loc_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split()
    assert len(first_line) > 5
    assert first_line[0] == "3110"
    assert int(first_line[1]) == 240
