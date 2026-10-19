#!/usr/bin/env bash
# ==============================================================================
# Master Experiment Campaign Runner (V54 -> V57) for Ubuntu Linux (RTX 5070 Ti)
#
# Usage:
#   ./scripts/run_all_v54_to_v57_ubuntu.sh proposed     # Run proposed models of V54, V55, V56, V57
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v57          # Run all V57 experiments
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v56          # Run all V56 experiments
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v55          # Run all V55 experiments
#   ./scripts/run_all_v54_to_v57_ubuntu.sh v54          # Run all V54 experiments
#   ./scripts/run_all_v54_to_v57_ubuntu.sh all          # Run the complete 18-experiment suite
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
        echo "Executing Proposed Flagship models across V54 -> V57..."
        bash scripts/run_v54_ubuntu.sh proposed
        bash scripts/run_v55_ubuntu.sh proposed
        bash scripts/run_v56_ubuntu.sh proposed
        bash scripts/run_v57_ubuntu.sh proposed
        ;;
    "all")
        echo "Executing full 18-experiment campaign across V54 -> V57..."
        bash scripts/run_v54_ubuntu.sh sequential
        bash scripts/run_v55_ubuntu.sh sequential
        bash scripts/run_v56_ubuntu.sh sequential
        bash scripts/run_v57_ubuntu.sh sequential
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [v54|v55|v56|v57|proposed|all]"
        exit 1
        ;;
esac

echo ""
echo "Master campaign execution for '$TARGET' completed successfully!"
