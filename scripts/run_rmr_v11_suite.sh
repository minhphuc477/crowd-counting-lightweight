#!/usr/bin/env bash
# ==============================================================================
# RMR-v11 Comprehensive Multi-Model Experiment Suite for Linux / Bash
# ==============================================================================

set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED="1"

PYTHON_EXE="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
fi

echo "================================================================================"
echo "  RMR-v11 MULTI-MODEL EXPERIMENT SUITE (Bash)"
echo "  Python: ${PYTHON_EXE}"
echo "  Date:   $(date)"
echo "================================================================================"

LOG_DIR="runs/sha_a/suite_logs_v11"
mkdir -p "${LOG_DIR}"

declare -A RUN_CONFIGS=(
    ["rmr_v11_canonical_dsr"]="configs/rmr_v11/rmr_v11_canonical_dsr.yaml"
    ["rmr_v11_ablation_no_curvature"]="configs/rmr_v11/rmr_v11_ablation_no_curvature.yaml"
    ["rmr_v11_ablation_no_trust_region"]="configs/rmr_v11/rmr_v11_ablation_no_trust_region.yaml"
    ["rmr_v11_ablation_no_hard_bg"]="configs/rmr_v11/rmr_v11_ablation_no_hard_bg.yaml"
    ["rmr_v11_ablation_no_fg_gate"]="configs/rmr_v11/rmr_v11_ablation_no_fg_gate.yaml"
    ["rmr_v11_control_no_solver"]="configs/rmr_v11/rmr_v11_control_no_solver.yaml"
)

ORDERED_RUNS=(
    "rmr_v11_canonical_dsr"
    "rmr_v11_ablation_no_curvature"
    "rmr_v11_ablation_no_trust_region"
    "rmr_v11_ablation_no_hard_bg"
    "rmr_v11_ablation_no_fg_gate"
    "rmr_v11_control_no_solver"
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
    run_model "${SINGLE_RUN}"
    exit 0
fi

for run_id in "${ORDERED_RUNS[@]}"; do
    run_model "${run_id}"
done

echo "================================================================================"
echo "  ALL RMR-v11 SUITE RUNS FINISHED"
echo "================================================================================"

${PYTHON_EXE} scripts/summarize_rmr_v11_suite.py
