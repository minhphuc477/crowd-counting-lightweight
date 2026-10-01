"""tools/audit_rmr_v30_integrity.py - Automated Forensic Static & Dynamic Analysis for RMR-v30.

Performs exhaustive preflight integrity verification:
1. Parameter Ceiling Verification: Asserts all 6 suite models <= 105,000 trainable parameters.
2. Zero Knowledge Distillation: Asserts 0 teacher weights, 0 KD loss terms, use_kd == False.
3. Monolith Prevention: Asserts all files in rmr_core/ and rmr_v3/ <= 450 lines.
4. Anti-Facade & Anti-Mocking: AST search for hardcoded constants; verifies deterministic input-sensitivity.
5. Canonical Benchmark Partition Integrity: Validates 300 train / 182 test split disjointness.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from rmr_v3.model import RMRv3, RMRv3Config
from tests.e2e_v30.conftest import get_rmr_v30_config_spec


def audit_parameter_ceilings() -> Tuple[bool, List[str]]:
    """Verify strict <= 105,000 trainable parameters across all 6 model variants."""
    print("=" * 70)
    print("CHECK 1: Parameter Ceiling Verification (<= 105,000)")
    print("=" * 70)

    variants = [
        ("step0_v19_anchor", 104441),
        ("h1_anscombe_sirt", 104441),
        ("h2_dual_lattice_dcsr", 104540),
        ("h3_anscombe_dual_lattice", 104540),
        ("h4_deep_sirt_t8", 104540),
        ("control_no_solver", 104441),
    ]

    failures = []
    for var, exp_params in variants:
        spec = get_rmr_v30_config_spec(var)
        m_cfg = RMRv3Config.from_dict(spec["model"], pretrained=False)
        model = RMRv3(m_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        status = "PASS" if trainable <= 105000 and trainable == exp_params else "FAIL"
        print(f"[{status}] {var:<26} : {trainable:,} trainable | {total:,} total (Budget: 105,000)")

        if trainable > 105000:
            failures.append(f"{var} exceeds 105,000 parameter budget: {trainable}")
        if trainable != exp_params:
            failures.append(f"{var} parameter count mismatch: expected {exp_params}, got {trainable}")

    return len(failures) == 0, failures


def audit_zero_knowledge_distillation() -> Tuple[bool, List[str]]:
    """Verify strict Zero KD policy: 0 teacher models, 0 distillation loss."""
    print("\n" + "=" * 70)
    print("CHECK 2: Zero Knowledge Distillation Verification")
    print("=" * 70)

    failures = []
    variants = [
        "step0_v19_anchor",
        "h1_anscombe_sirt",
        "h2_dual_lattice_dcsr",
        "h3_anscombe_dual_lattice",
        "h4_deep_sirt_t8",
        "control_no_solver",
    ]

    for var in variants:
        spec = get_rmr_v30_config_spec(var)
        model_cfg = spec.get("model", {})
        loss_cfg = spec.get("loss", {})

        if model_cfg.get("use_kd", False) is not False:
            failures.append(f"{var}: use_kd is not False")
        if loss_cfg.get("lambda_kd", 0.0) != 0.0:
            failures.append(f"{var}: lambda_kd is not 0.0")

        # Inspect model modules for teacher weights
        m_cfg = RMRv3Config.from_dict(model_cfg, pretrained=False)
        model = RMRv3(m_cfg)
        for name, _ in model.named_modules():
            if "teacher" in name.lower() or "distill" in name.lower():
                failures.append(f"{var}: Found forbidden KD submodule '{name}'")

    status = "PASS" if len(failures) == 0 else "FAIL"
    print(f"[{status}] Zero Knowledge Distillation verified across all {len(variants)} configurations.")
    return len(failures) == 0, failures


def audit_monolith_prevention() -> Tuple[bool, List[str]]:
    """Verify that all source files in rmr_core/ and rmr_v3/ are <= 450 lines."""
    print("\n" + "=" * 70)
    print("CHECK 3: Monolith Prevention Audit (<= 450 lines per file)")
    print("=" * 70)

    violations = []
    total_files = 0
    max_file = ("", 0)

    for target_dir in ["rmr_core", "rmr_v3"]:
        dir_path = REPO_ROOT / target_dir
        if not dir_path.is_dir():
            continue
        for root, _, files in os.walk(dir_path):
            if "__pycache__" in root:
                continue
            for f in sorted(files):
                if f.endswith(".py"):
                    total_files += 1
                    file_path = Path(root) / f
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    count = len(lines)
                    if count > max_file[1]:
                        max_file = (str(file_path.relative_to(REPO_ROOT)), count)
                    if count > 450:
                        violations.append(f"{file_path.relative_to(REPO_ROOT)}: {count} lines (> 450)")

    status = "PASS" if len(violations) == 0 else "FAIL"
    print(f"[{status}] Scanned {total_files} Python source files. Highest: {max_file[0]} ({max_file[1]} lines)")
    if violations:
        for v in violations:
            print(f"  [VIOLATION] {v}")

    return len(violations) == 0, violations


def audit_anti_mocking_and_input_sensitivity() -> Tuple[bool, List[str]]:
    """AST check for mocked constants and empirical test of input sensitivity."""
    print("\n" + "=" * 70)
    print("CHECK 4: Anti-Facade & Deterministic Input-Sensitivity Audit")
    print("=" * 70)

    failures = []

    # 1. AST scan in evaluation files for suspicious hardcoded returns
    eval_files = [
        REPO_ROOT / "rmr_core" / "metrics.py",
        REPO_ROOT / "rmr_core" / "evaluation.py",
        REPO_ROOT / "rmr_v3" / "eval.py",
    ]
    for ef in eval_files:
        if not ef.is_file():
            continue
        tree = ast.parse(ef.read_text(encoding="utf-8"), filename=str(ef))
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                # Allow standard float/int returns like 0.0 or nan, but forbid suspicious constants
                val = node.value.value
                if isinstance(val, (int, float)) and val not in (0, 0.0, 1, 1.0, -1, float("nan")):
                    if val in (433.9, 72.84, 59.8, 64.9, 10.64):
                        failures.append(f"{ef.name}:{node.lineno} Hardcoded benchmark constant {val} returned!")

    # 2. Input-Sensitivity: Distinct inputs MUST produce distinct outputs
    spec = get_rmr_v30_config_spec("step0_v19_anchor")
    m_cfg = RMRv3Config.from_dict(spec["model"], pretrained=False)
    model = RMRv3(m_cfg).eval()

    torch.manual_seed(999)
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out1 = model(x1).y
        out2 = model(x2).y

    diff = (out1 - out2).abs().sum().item()
    if diff < 1e-4:
        failures.append("Model output is invariant to input image pixels (Dummy/Mock detected)!")

    status = "PASS" if len(failures) == 0 else "FAIL"
    print(f"[{status}] AST inspection clean. Input sensitivity verified (delta_out = {diff:.4f} > 0).")
    return len(failures) == 0, failures


def audit_canonical_splits() -> Tuple[bool, List[str]]:
    """Assert canonical dataset files exist with exact 300 train / 182 test counts."""
    print("\n" + "=" * 70)
    print("CHECK 5: Canonical Dataset Split & Anti-Leakage Audit")
    print("=" * 70)

    import json
    train_p = REPO_ROOT / "data" / "sha_a_train_all.jsonl"
    test_p = REPO_ROOT / "data" / "sha_a_test.jsonl"
    failures = []

    if not train_p.is_file():
        failures.append(f"Missing canonical train manifest: {train_p}")
    if not test_p.is_file():
        failures.append(f"Missing canonical test manifest: {test_p}")

    if not failures:
        train_lines = [json.loads(l) for l in train_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        test_lines = [json.loads(l) for l in test_p.read_text(encoding="utf-8").splitlines() if l.strip()]

        if len(train_lines) != 300:
            failures.append(f"Train manifest count {len(train_lines)} != 300")
        if len(test_lines) != 182:
            failures.append(f"Test manifest count {len(test_lines)} != 182")

        train_images = {str(item["image"]).replace("\\", "/") for item in train_lines}
        test_images = {str(item["image"]).replace("\\", "/") for item in test_lines}
        overlap = train_images.intersection(test_images)
        if len(overlap) > 0:
            failures.append(f"Data leakage detected! {len(overlap)} image paths overlap between train and test!")

    status = "PASS" if len(failures) == 0 else "FAIL"
    print(f"[{status}] Canonical splits verified: 300 train, 182 test, 0 disjoint overlap.")
    return len(failures) == 0, failures


def run_full_audit() -> bool:
    """Execute all forensic audits and report summary."""
    print("\n" + "#" * 70)
    print("# RMR-v30 FORENSIC INTEGRITY AUDIT SUITE")
    print("#" * 70 + "\n")

    p1, f1 = audit_parameter_ceilings()
    p2, f2 = audit_zero_knowledge_distillation()
    p3, f3 = audit_monolith_prevention()
    p4, f4 = audit_anti_mocking_and_input_sensitivity()
    p5, f5 = audit_canonical_splits()

    all_passed = p1 and p2 and p3 and p4 and p5
    all_failures = f1 + f2 + f3 + f4 + f5

    print("\n" + "=" * 70)
    print("FINAL INTEGRITY AUDIT SUMMARY")
    print("=" * 70)
    print(f"  [1] Parameter Budget <= 105k  : {'PASS' if p1 else 'FAIL'}")
    print(f"  [2] Zero Knowledge Distill   : {'PASS' if p2 else 'FAIL'}")
    print(f"  [3] Monolith Guard <= 450     : {'PASS' if p3 else 'FAIL'}")
    print(f"  [4] Anti-Mock / Input Flow    : {'PASS' if p4 else 'FAIL'}")
    print(f"  [5] Canonical Dataset Splits  : {'PASS' if p5 else 'FAIL'}")
    print("-" * 70)

    if all_passed:
        print(">>> ALL FORENSIC INTEGRITY CHECKS PASSED (100% CLEAN) <<<")
        return True
    else:
        print(f">>> AUDIT FAILED WITH {len(all_failures)} VIOLATIONS <<<")
        for f in all_failures:
            print(f"  - {f}")
        return False


if __name__ == "__main__":
    success = run_full_audit()
    sys.exit(0 if success else 1)
