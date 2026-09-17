#!/usr/bin/env bash
# Bash Runner for RMR-v22 Sub-60 Experiment Suite
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v22_canonical:configs/rmr_v22/rmr_v22_canonical.yaml"
    "rmr_v22_ablation_no_scale_cond_head:configs/rmr_v22/rmr_v22_ablation_no_scale_cond_head.yaml"
    "rmr_v22_ablation_no_density_gated_diffusion:configs/rmr_v22/rmr_v22_ablation_no_density_gated_diffusion.yaml"
    "rmr_v22_control_no_solver:configs/rmr_v22/rmr_v22_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v22"
mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "  RMR-v22 SUB-60 EXPERIMENT SUITE"
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
        "${PYTHON_EXE}" -m rmr_v3.eval --checkpoint "${CKPT}" --tta 2>&1 | tee "${EVAL_LOG}"
    fi
done

echo ""
echo "All RMR-v22 experiments completed!"
