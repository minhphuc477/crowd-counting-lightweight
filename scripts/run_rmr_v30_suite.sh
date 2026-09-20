#!/usr/bin/env bash
# ==============================================================================
# RMR-v30 Hypothesis Ladder Benchmark Suite (Linux / Bash / HPC)
# Target: Break the Sub-60 MAE barrier on ShanghaiTech Part A
# Gold Baseline: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)
# ==============================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v30_step0_v19_anchor:configs/rmr_v30/rmr_v30_step0_v19_anchor.yaml"
    "rmr_v30_h1_anscombe_sirt:configs/rmr_v30/rmr_v30_h1_anscombe_sirt.yaml"
    "rmr_v30_h2_dual_lattice_dcsr:configs/rmr_v30/rmr_v30_h2_dual_lattice_dcsr.yaml"
    "rmr_v30_h3_anscombe_dual_lattice:configs/rmr_v30/rmr_v30_h3_anscombe_dual_lattice.yaml"
    "rmr_v30_h4_deep_sirt_t8:configs/rmr_v30/rmr_v30_h4_deep_sirt_t8.yaml"
    "rmr_v30_control_no_solver:configs/rmr_v30/rmr_v30_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v30"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR-v30 HYPOTHESIS LADDER EXPERIMENT SUITE (Bash)"
echo "  Gold Reference: RMR-v19 Canonical Isotropic (72.61 TTA MAE, 72.84 Direct MAE)"
echo "  Target: Sub-60 MAE (< 60.0)"
echo "================================================================================"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
fi

for ENTRY in "${MODELS[@]}"; do
    RUN_ID="${ENTRY%%:*}"
    CFG="${ENTRY##*:}"
    LOG_FILE="$LOG_DIR/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: $RUN_ID ($CFG)"
    $PYTHON_BIN -m rmr_v3.train --config "$CFG" --run-id "$RUN_ID" --overwrite 2>&1 | tee "$LOG_FILE"
    echo ">>> Completed: $RUN_ID"

    CKPT_PATH="runs/sha_a/${RUN_ID}/best_val_mae.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating with Horizontal Flip TTA: $CKPT_PATH"
        EVAL_LOG="$LOG_DIR/${RUN_ID}_eval_tta.log"
        $PYTHON_BIN -m rmr_v3.eval --checkpoint "$CKPT_PATH" --tta --no-tiling 2>&1 | tee "$EVAL_LOG"
    fi
done

echo ""
echo ">>> Summarizing RMR-v30 Suite Results:"
$PYTHON_BIN scripts/summarize_rmr_v30_suite.py

echo ""
echo "All RMR-v30 experiments completed!"
