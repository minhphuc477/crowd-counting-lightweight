#!/usr/bin/env bash
# ==============================================================================
# Runner for Gen 10 Anti-Regression Breakthrough Suite (Target: MAE < 60.0)
# Baseline: sub60_e5 Champion (71.51 MAE / 110.80 RMSE)
# Invariants: <= 105,000 params, Zero KD, Canonical ShanghaiTech Part A
# Delicate Equilibrium Intact: curvature=-8.0, hurdle=product, lambda_cell=0.0
# ==============================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

declare -a MODELS=(
    # 0. Canonical Anchor: Exact sub60_e5 Reproduction Baseline (104,441 params)
    "g10_anchor_sub60_e5:configs/rmr_research/gen10/g10_anchor_sub60_e5.yaml"

    # 1. Structural Resolution: Stride 2 Sub-pixel Dirichlet-Multinomial (104,893 params)
    "g10_subpixel_dm_stride2:configs/rmr_research/gen10/g10_subpixel_dm_stride2.yaml"

    # 2. Deadband Calibration: Clean Isolated Asymmetric Morozov (104,441 params)
    "g10_clean_asym_morozov:configs/rmr_research/gen10/g10_clean_asym_morozov.yaml"

    # 3. Frequency Regularization: Calibrated ChfL lambda=0.35 (104,441 params)
    "g10_chfl_calibrated_lam035:configs/rmr_research/gen10/g10_chfl_calibrated_lam035.yaml"

    # 4. Architectural Control: Stride 2 Subpixel Without Solver (104,893 params)
    "g10_subpixel_stride2_no_solver:configs/rmr_research/gen10/g10_subpixel_stride2_no_solver.yaml"

    # 5. Spatial + Solver Synergy: Stride 2 Subpixel + Asymmetric Morozov (104,893 params)
    "g10_subpixel_stride2_asym:configs/rmr_research/gen10/g10_subpixel_stride2_asym.yaml"

    # 6. Spatial + Frequency Synergy: Stride 2 Subpixel + Calibrated ChfL (104,893 params)
    "g10_subpixel_stride2_chfl:configs/rmr_research/gen10/g10_subpixel_stride2_chfl.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_gen10"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR GEN 10 ANTI-REGRESSION BREAKTHROUGH SUITE (Bash)"
echo "  7 Targeted Experiments | Complete Causal Isolation & Synthesis"
echo "  Target: Breakthrough Sub-60 MAE | Strict Budget: <= 105k Params, Zero KD"
echo "================================================================================"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
fi

for ENTRY in "${MODELS[@]}"; do
    RUN_ID="${ENTRY%%:*}"
    CFG="${ENTRY##*:}"
    LOG_FILE="$LOG_DIR/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: $RUN_ID ($CFG)"
    $PYTHON_BIN -m rmr_v3.train \
        --config "$CFG" \
        --run-id "$RUN_ID" \
        --workers 0 \
        --num-threads 2 \
        --no-cudnn-benchmark \
        --overwrite 2>&1 | tee "$LOG_FILE"
    echo ">>> Completed training: $RUN_ID"

    CKPT_PATH="runs/sha_a/$RUN_ID/best_val_mae.pt"
    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating Standard Test: $CKPT_PATH"
        $PYTHON_BIN -m rmr_v3.evaluate \
            --checkpoint "$CKPT_PATH" \
            --manifest data/sha_a_test.jsonl \
            --crop-size 512 \
            --tta false \
            --output-json "runs/sha_a/$RUN_ID/eval_test_canonical.json" || true
    fi
done

echo ""
echo "================================================================================"
echo "  GEN 10 EXPERIMENT SUITE COMPLETED!"
echo "================================================================================"
