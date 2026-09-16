#!/usr/bin/env bash
# ==============================================================================
# RMR-v20 Comprehensive Multi-Model Experiment Suite for Linux / Bash
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
fi

echo "================================================================================"
echo "  RMR-v20 SUB-60 EXPERIMENT SUITE (Bash)"
echo "  Python: ${PYTHON_EXE}"
echo "  Date:   $(date)"
echo "================================================================================"

LOG_DIR="runs/sha_a/suite_logs_v20"
mkdir -p "${LOG_DIR}"

declare -A RUN_CONFIGS=(
    ["rmr_v20_canonical"]="configs/rmr_v20/rmr_v20_canonical.yaml"
    ["rmr_v20_ablation_no_coord_attn"]="configs/rmr_v20/rmr_v20_ablation_no_coord_attn.yaml"
    ["rmr_v20_ablation_no_nesterov"]="configs/rmr_v20/rmr_v20_ablation_no_nesterov.yaml"
    ["rmr_v20_ablation_no_adaptive_relax"]="configs/rmr_v20/rmr_v20_ablation_no_adaptive_relax.yaml"
    ["rmr_v20_ablation_no_dense_loss"]="configs/rmr_v20/rmr_v20_ablation_no_dense_loss.yaml"
    ["rmr_v20_control_no_solver"]="configs/rmr_v20/rmr_v20_control_no_solver.yaml"
)

ORDERED_RUNS=(
    "rmr_v20_canonical"
    "rmr_v20_ablation_no_coord_attn"
    "rmr_v20_ablation_no_nesterov"
    "rmr_v20_ablation_no_adaptive_relax"
    "rmr_v20_ablation_no_dense_loss"
    "rmr_v20_control_no_solver"
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
}

if [ -n "${SINGLE_RUN}" ]; then
    if [[ -z "${RUN_CONFIGS[$SINGLE_RUN]}" ]]; then
        echo "Error: Unknown run ID '${SINGLE_RUN}'."
        echo "Available runs: ${ORDERED_RUNS[*]}"
        exit 1
    fi
    run_model "${SINGLE_RUN}"
else
    echo "Running full suite sequentially (${#ORDERED_RUNS[@]} runs)..."
    for r in "${ORDERED_RUNS[@]}"; do
        run_model "$r"
    done
fi

echo "================================================================================"
echo "  ALL REQUESTED RMR-V20 RUNS COMPLETED SUCCESSFULLY"
echo "================================================================================"
