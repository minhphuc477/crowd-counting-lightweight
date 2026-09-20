"""RMR-v30 4-Tier Opaque-Box End-to-End & Unit Test Suite.

Exhaustive multi-tier verification covering:
- Tier 1: Nominal Feature Coverage & Mathematical Oracles (F1-F7, >= 35 tests)
- Tier 2: Boundary, Singularities & Edge Cases (>= 35 tests)
- Tier 3: Cross-Feature Combinations & Unrolled Solver Dynamics (>= 8 tests)
- Tier 4: Canonical ShanghaiTech Part A Benchmarks & Evaluation Scenarios (>= 7 tests)

Integrity constraints enforced:
- Maximum trainable parameters <= 105,000 (RMR-v30 target <= 104,800)
- Zero Knowledge Distillation (zero teacher weights, zero KD loss)
- Strict Monolith Prevention (all source files <= 450 lines)
- Mathematical Invariants (Mass conservation, support invariance, Anscombe asymptotic variance)
"""
from __future__ import annotations

__version__ = "30.0.0"
__all__ = ["__version__"]
