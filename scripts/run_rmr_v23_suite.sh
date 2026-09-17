#!/usr/bin/env bash
# Bash Runner for RMR-v23 Sub-60 Experiment Suite
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

declare -a MODELS=(
    "rmr_v23_canonical:configs/rmr_v23/rmr_v23_canonical.yaml"
    "rmr_v23_ablation_no_bb:configs/rmr_v23/rmr_v23_ablation_no_bb.yaml"
    "rmr_v23_ablation_no_density_gated_diffusion:configs/rmr_v23/rmr_v23_ablation_no_density_gated_diffusion.yaml"
    "rmr_v23_ablation_no_gated_curvature:configs/rmr_v23/rmr_v23_ablation_no_gated_curvature.yaml"
    "rmr_v23_ablation_no_scale_align:configs/rmr_v23/rmr_v23_ablation_no_scale_align.yaml"
    "rmr_v23_control_no_solver:configs/rmr_v23/rmr_v23_control_no_solver.yaml"
)

LOG_DIR="runs/sha_a/suite_logs_v23"
mkdir -p "${LOG_DIR}"

echo "================================================================================"
echo "  RMR-v23 SUB-60 EXPERIMENT SUITE"
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
echo "All RMR-v23 experiments completed!"
