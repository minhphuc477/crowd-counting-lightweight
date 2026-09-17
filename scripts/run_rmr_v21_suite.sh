#!/usr/bin/env bash
# ==============================================================================
# RMR-v21 Breakthrough Suite Runner: Breaking Sub-60 MAE with Hybrid Flux
# ==============================================================================

set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED="1"

PYTHON_EXE="python"
if command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
fi
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_EXE=".venv/Scripts/python.exe"
fi

echo "================================================================================"
echo "  RMR-v21 BREAKTHROUGH EXPERIMENT SUITE (Bash)"
echo "  Python: ${PYTHON_EXE}"
echo "  Date:   $(date)"
echo "================================================================================"

LOG_DIR="runs/sha_a/suite_logs_v21"
mkdir -p "${LOG_DIR}"

declare -A RUN_CONFIGS=(
    ["rmr_v21_canonical"]="configs/rmr_v21/rmr_v21_canonical.yaml"
    ["rmr_v21_ablation_no_hybrid_flux"]="configs/rmr_v21/rmr_v21_ablation_no_hybrid_flux.yaml"
    ["rmr_v21_ablation_no_bb_step"]="configs/rmr_v21/rmr_v21_ablation_no_bb_step.yaml"
    ["rmr_v21_ablation_no_sample_loss"]="configs/rmr_v21/rmr_v21_ablation_no_sample_loss.yaml"
    ["rmr_v21_ablation_no_curvature"]="configs/rmr_v21/rmr_v21_ablation_no_curvature.yaml"
    ["rmr_v21_control_no_solver"]="configs/rmr_v21/rmr_v21_control_no_solver.yaml"
)

ORDERED_RUNS=(
    "rmr_v21_canonical"
    "rmr_v21_ablation_no_hybrid_flux"
    "rmr_v21_ablation_no_bb_step"
    "rmr_v21_ablation_no_sample_loss"
    "rmr_v21_ablation_no_curvature"
    "rmr_v21_control_no_solver"
)

SINGLE_RUN="${1:-}"

run_model() {
    local run_id="$1"
    local cfg_file="${RUN_CONFIGS[$run_id]}"
    local log_file="${LOG_DIR}/${run_id}.log"

    echo "================================================================================"
    echo "  STARTING: ${run_id}"
    echo "  Config:   ${cfg_file}"
    echo "  Log:      ${log_file}"
    echo "================================================================================"

    ${PYTHON_EXE} -m rmr_v3.train --config "${cfg_file}" --run-id "${run_id}" 2>&1 | tee "${log_file}"
    echo "FINISHED: ${run_id}"

    # Evaluate best checkpoint with TTA
    local ckpt_path="runs/sha_a/${run_id}/best_val_mae.pt"
    if [ -f "${ckpt_path}" ]; then
        echo ">>> Evaluating with Horizontal Flip TTA: ${ckpt_path}"
        ${PYTHON_EXE} -m rmr_v3.eval --checkpoint "${ckpt_path}" --tta 2>&1 | tee "${LOG_DIR}/${run_id}_eval_tta.log"
    fi
}

if [ -n "${SINGLE_RUN}" ]; then
    if [ -z "${RUN_CONFIGS[$SINGLE_RUN]+x}" ]; then
        echo "ERROR: Unknown run identifier '${SINGLE_RUN}'."
        echo "Available runs: ${ORDERED_RUNS[*]}"
        exit 1
    fi
    run_model "${SINGLE_RUN}"
else
    for run_id in "${ORDERED_RUNS[@]}"; do
        run_model "${run_id}"
    done
fi

echo "================================================================================"
echo "  RMR-v21 SUITE COMPLETE!"
echo "================================================================================"
