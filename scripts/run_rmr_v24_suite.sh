#!/usr/bin/env bash
# Bash Runner for RMR-v24 Sub-60 Experiment Suite
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v24_canonical:configs/rmr_v24/rmr_v24_canonical.yaml"
    "rmr_v24_ablation_no_abb:configs/rmr_v24/rmr_v24_ablation_no_abb.yaml"
    "rmr_v24_ablation_no_perspective:configs/rmr_v24/rmr_v24_ablation_no_perspective.yaml"
    "rmr_v24_ablation_no_density_gated_diffusion:configs/rmr_v24/rmr_v24_ablation_no_density_gated_diffusion.yaml"
    "rmr_v24_ablation_no_curvature:configs/rmr_v24/rmr_v24_ablation_no_curvature.yaml"
    "rmr_v24_control_no_solver:configs/rmr_v24/rmr_v24_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v24"
mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "  RMR-v24 SUB-60 EXPERIMENT SUITE"
echo "================================================================================"

PYTHON_EXE="python"
if command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
fi
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
fi

for entry in "${MODELS[@]}"; do
    RUN_ID="${entry%%:*}"
    CFG="${entry##*:}"
    LOG="${LOG_DIR}/${RUN_ID}.log"

    echo ""
    echo ">>> Starting training for: ${RUN_ID} (${CFG})"
    "${PYTHON_EXE}" -m rmr_v3.train --config "${CFG}" --run-id "${RUN_ID}" 2>&1 | tee "${LOG}"
    echo ">>> Completed: ${RUN_ID}"

    CKPT="runs/sha_a/${RUN_ID}/best_val_mae.pt"
    if [[ -f "${CKPT}" ]]; then
        echo ">>> Evaluating with Horizontal Flip TTA: ${CKPT}"
        EVAL_LOG="${LOG_DIR}/${RUN_ID}_eval_tta.log"
        "${PYTHON_EXE}" -m rmr_v3.eval --checkpoint "${CKPT}" --tta --no-tiling 2>&1 | tee "${EVAL_LOG}"
    fi
done

echo ""
echo ">>> Summarizing RMR-v24 Suite Results:"
"${PYTHON_EXE}" scripts/summarize_rmr_v24_suite.py

echo ""
echo "All RMR-v24 experiments completed!"
