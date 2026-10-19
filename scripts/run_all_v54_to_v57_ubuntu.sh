#!/usr/bin/env bash
# ==============================================================================
# Master Experiment Campaign Runner (V54 -> V57) for Ubuntu Linux (RTX 5070 Ti)
#
# Usage:
#   ./scripts/run_all_v54_to_v57_ubuntu.sh proposed            # Run proposed models (V54->V57) sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh ablation            # Run all ablations (V54->V57) sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh parallel-proposed   # Run proposed models (V54->V57) in parallel waves
#   ./scripts/run_all_v54_to_v57_ubuntu.sh parallel-ablation   # Run all ablations (V54->V57) in parallel waves
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v57                 # Run all V57 experiments sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v56                 # Run all V56 experiments sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v55                 # Run all V55 experiments sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v54                 # Run all V54 experiments sequentially
#   ./scripts/run_all_v54_to_v57_ubuntu.sh all                 # Run the complete 18-experiment suite sequentially
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TARGET="${1:-v57}"

echo "================================================================================"
echo "  RMR MASTER CAMPAIGN LAUNCHER: V54 -> V57"
echo "  Target: $TARGET"
echo "  Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

case "$TARGET" in
    "v54")
        bash scripts/run_v54_ubuntu.sh sequential
        ;;
    "v55")
        bash scripts/run_v55_ubuntu.sh sequential
        ;;
    "v56")
        bash scripts/run_v56_ubuntu.sh sequential
        ;;
    "v57")
        bash scripts/run_v57_ubuntu.sh sequential
        ;;
    "proposed")
        echo "Executing Proposed Flagship models across V54 -> V57 sequentially..."
        bash scripts/run_v54_ubuntu.sh proposed
        bash scripts/run_v55_ubuntu.sh proposed
        bash scripts/run_v56_ubuntu.sh proposed
        bash scripts/run_v57_ubuntu.sh proposed
        ;;
    "ablation")
        echo "Executing All Ablation & Control models across V54 -> V57 sequentially..."
        echo ">>> Stage 1/4: V54 Ablations (3 models)..."
        bash scripts/run_v54_ubuntu.sh ablation
        echo ">>> Stage 2/4: V55 Ablations (3 models)..."
        bash scripts/run_v55_ubuntu.sh ablation
        echo ">>> Stage 3/4: V56 Ablations (2 models)..."
        bash scripts/run_v56_ubuntu.sh ablation
        echo ">>> Stage 4/4: V57 Ablations (2 models)..."
        bash scripts/run_v57_ubuntu.sh ablation
        ;;
    "parallel-proposed")
        echo "Executing Proposed Flagship models across V54 -> V57 in 2-way parallel waves..."
        bash scripts/run_v54_ubuntu.sh parallel-proposed
        bash scripts/run_v55_ubuntu.sh parallel-proposed
        bash scripts/run_v56_ubuntu.sh parallel-proposed
        bash scripts/run_v57_ubuntu.sh parallel-proposed
        ;;
    "parallel-ablation")
        echo "Executing All Ablation & Control models across V54 -> V57 in 2-way parallel waves..."
        bash scripts/run_v54_ubuntu.sh parallel-ablation
        bash scripts/run_v55_ubuntu.sh parallel-ablation
        bash scripts/run_v56_ubuntu.sh parallel-ablation
        bash scripts/run_v57_ubuntu.sh parallel-ablation
        ;;
    "all")
        echo "Executing full 18-experiment campaign across V54 -> V57 sequentially..."
        bash scripts/run_v54_ubuntu.sh sequential
        bash scripts/run_v55_ubuntu.sh sequential
        bash scripts/run_v56_ubuntu.sh sequential
        bash scripts/run_v57_ubuntu.sh sequential
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [v54|v55|v56|v57|proposed|ablation|parallel-proposed|parallel-ablation|all]"
        exit 1
        ;;
esac

echo ""
echo "Master campaign execution for '$TARGET' completed successfully!"
