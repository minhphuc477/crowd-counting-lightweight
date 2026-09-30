#!/usr/bin/env bash
# Bash Runner for Gen 9 Mathematical Fixes & Breakthrough Ablation Suite
# 6 Experiments | Target: Breakthrough Sub-60 MAE (< 60.0)
# Baseline: sub60_e5 MAE Champion (71.51 MAE)
# Hard Constraints: <= 105,000 params, Zero KD, ShanghaiTech Part A (300 train / 182 test)
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

declare -a MODELS=(
    # 1. Bug Fix 1: curvature_alpha_init = 0.0 (active from epoch 1)
    "g9_fix_curvature_init:configs/rmr_research/gen9/g9_fix_curvature_init.yaml"

    # 2. Bug Fix 2: hurdle_gating_mode = occupancy (eliminates 15-25% dense mass erosion)
    "g9_fix_hurdle_occupancy:configs/rmr_research/gen9/g9_fix_hurdle_occupancy.yaml"

    # 3. Asymmetric Morozov Discrepancy (gamma_under = 0.35 bias correction)
    "g9_asym_morozov_e5:configs/rmr_research/gen9/g9_asym_morozov_e5.yaml"

    # 4. Softer MCP Firm Threshold (proximal_mu = 1.5)
    "g9_softer_firm_mu15:configs/rmr_research/gen9/g9_softer_firm_mu15.yaml"

    # 5. DCT-II Neumann Spectral Loss on corrected baseline
    "g9_dct_spectral_fixed:configs/rmr_research/gen9/g9_dct_spectral_fixed.yaml"

    # 6. Compound Best: All 4 orthogonal fixes synthesized
    "g9_compound_all_fixes:configs/rmr_research/gen9/g9_compound_all_fixes.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_gen9"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "  RMR GEN 9 COMPREHENSIVE EXPERIMENT SUITE (Bash)"
echo "  6 Experiments | Target: Sub-60 MAE (< 60.0)"
echo "  Baseline: sub60_e5 (71.51 MAE) | Constraints: <= 105k params, Zero KD"
echo "================================================================================"

PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

for ENTRY in "${MODELS[@]}"; do
    RUN_ID="${ENTRY%%:*}"
    CFG="${ENTRY##*:}"
    LOG_FILE="$LOG_DIR/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: $RUN_ID ($CFG)"
    $PYTHON_BIN -m rmr_v3.train --config "$CFG" --run-id "$RUN_ID" --overwrite 2>&1 | tee "$LOG_FILE"
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
echo "  GEN 9 EXPERIMENT SUITE COMPLETED!"
echo "================================================================================"
