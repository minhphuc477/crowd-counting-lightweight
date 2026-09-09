#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Parallel 4-Model Training & Evaluation Runner for Linux/Ubuntu (16GB VRAM)
# Runs all 4 models simultaneously on GPU:
#   1. V3-A: Probabilistic Uniform Control (configs/rmr_v3/probabilistic_uniform.yaml)
#   2. V3-B: Reliability Weighted - Proposed (configs/rmr_v3/reliability_weighted.yaml)
#   3. B5-P: Canonical Deterministic Baseline (configs/rmr_v2/rmr_projected_t2.yaml)
#   4. V4:   RMR-v4 Full Candidate (configs/rmr_v4/final_candidate.yaml)
# ==============================================================================

# Prevent CUDA memory fragmentation under multi-process training
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

LOG_DIR="runs/sha_a/parallel_logs"
mkdir -p "$LOG_DIR"

echo -e "\n================================================================================"
echo "  PARALLEL 4-MODEL EXPERIMENT RUNNER (16GB GPU Mode)"
echo "  Launching 4 models concurrently in background:"
echo "    [1] V3-A (Probabilistic Uniform Control)"
echo "    [2] V3-B (Reliability Weighted - Proposed)"
echo "    [3] B5-P (Deterministic Baseline)"
echo "    [4] V4-Candidate (Full Multiscale + Mean/Std + Native Pooling)"
echo "  Logs saved to: $LOG_DIR/"
echo -e "================================================================================\n"

# Helper to launch a single training job in background
launch_training() {
    local id="$1"
    local name="$2"
    local mod="$3"
    local cfg="$4"
    local out_dir="$5"
    local log_file="$LOG_DIR/${id}.log"

    local last_ckpt="$out_dir/last.pt"
    local train_args=("-m" "$mod" "--config" "$cfg")
    
    if [ -f "$last_ckpt" ]; then
        echo "  [$id: RESUME] Found checkpoint at $last_ckpt. Resuming..."
        train_args+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        train_args+=("--overwrite")
    fi

    echo "  [START] $name (log: $log_file)"
    "$PYTHON_EXE" "${train_args[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting all 4 training jobs concurrently..."

PID_V3A=$(launch_training "v3a" "V3-A Uniform Control" "rmr_v3.train" "configs/rmr_v3/probabilistic_uniform.yaml" "runs/sha_a/rmr_v3_uniform_control_seed42")
PID_V3B=$(launch_training "v3b" "V3-B Reliability Weighted" "rmr_v3.train" "configs/rmr_v3/reliability_weighted.yaml" "runs/sha_a/rmr_v3_rw_seed42")
PID_B5P=$(launch_training "b5p" "B5-P Canonical Baseline" "rmr_v2.train" "configs/rmr_v2/rmr_projected_t2.yaml" "runs/sha_a/b5p_canonical_seed42")
PID_V4=$(launch_training "v4_cand" "RMR-v4 Full Candidate" "rmr_v3.train" "configs/rmr_v4/final_candidate.yaml" "runs/sha_a/rmr_v4_candidate_seed42")

echo -e "\nAll 4 processes launched on GPU:"
echo "  PID V3-A:      $PID_V3A  (tail -f $LOG_DIR/v3a.log)"
echo "  PID V3-B:      $PID_V3B  (tail -f $LOG_DIR/v3b.log)"
echo "  PID B5-P:      $PID_B5P  (tail -f $LOG_DIR/b5p.log)"
echo "  PID V4-Cand:   $PID_V4   (tail -f $LOG_DIR/v4_cand.log)"

echo -e "\nWaiting for all 4 models to finish training..."
echo "Tip: You can inspect GPU usage in another terminal using: watch -n 1 nvidia-smi"

# Wait for all processes and capture exit statuses
wait $PID_V3A || echo "Warning: V3-A exited with code $?"
wait $PID_V3B || echo "Warning: V3-B exited with code $?"
wait $PID_B5P || echo "Warning: B5-P exited with code $?"
wait $PID_V4 || echo "Warning: V4-Candidate exited with code $?"

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] All 4 training jobs have finished!"
echo "================================================================================"
echo "  EVALUATING ALL 4 BEST CHECKPOINTS ON SHANGHAITECH PART A (182 IMAGES)..."
echo -e "================================================================================\n"

# Evaluate V3-A
echo "Evaluating V3-A..."
"$PYTHON_EXE" -m rmr_v3.eval \
    --checkpoint runs/sha_a/rmr_v3_uniform_control_seed42/best_val_mae.pt \
    --manifest data/sha_a_test.jsonl \
    --output-dir runs/sha_a/rmr_v3_uniform_control_seed42/eval_test

# Evaluate V3-B
echo "Evaluating V3-B..."
"$PYTHON_EXE" -m rmr_v3.eval \
    --checkpoint runs/sha_a/rmr_v3_rw_seed42/best_val_mae.pt \
    --manifest data/sha_a_test.jsonl \
    --output-dir runs/sha_a/rmr_v3_rw_seed42/eval_test

# Evaluate B5-P
echo "Evaluating B5-P..."
"$PYTHON_EXE" -m rmr_v2.eval \
    --checkpoint runs/sha_a/b5p_canonical_seed42/best_val_mae.pt \
    --manifest data/sha_a_test.jsonl \
    --output-dir runs/sha_a/b5p_canonical_seed42/eval_test

# Evaluate V4 Candidate
echo "Evaluating V4 Candidate..."
"$PYTHON_EXE" -m rmr_v3.eval \
    --checkpoint runs/sha_a/rmr_v4_candidate_seed42/best_val_mae.pt \
    --manifest data/sha_a_test.jsonl \
    --output-dir runs/sha_a/rmr_v4_candidate_seed42/eval_test

echo -e "\n================================================================================"
echo "  RUNNING PAIRED STATISTICAL COMPARISONS..."
echo -e "================================================================================\n"

PRED_V3A="runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
PRED_V3B="runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
PRED_B5P="runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"
PRED_V4="runs/sha_a/rmr_v4_candidate_seed42/eval_test/predictions.csv"

# 1. V3-A vs V3-B
echo "--- 1. Paired Comparison: V3-A (Uniform) vs V3-B (Reliability Weighted) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3A" "$PRED_V3B" \
    --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred \
    --output "runs/sha_a/comparison_v3a_vs_v3b.json"

# 2. B5-P vs V3-B
echo -e "\n--- 2. Paired Comparison: B5-P (Deterministic) vs V3-B (Reliability Weighted) ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_B5P" "$PRED_V3B" \
    --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred \
    --output "runs/sha_a/comparison_b5p_vs_v3b.json"

# 3. V3-B vs V4 Candidate
echo -e "\n--- 3. Paired Comparison: V3-B (RW-RMR) vs V4 Candidate ---"
"$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4" \
    --name-a "V3-B_RW" --name-b "V4_Candidate" --pred-col pred \
    --output "runs/sha_a/comparison_v3b_vs_v4_candidate.json"

echo -e "\n================================================================================"
echo "  PARALLEL 4-MODEL SUITE COMPLETED SUCCESSFULLY!"
echo -e "================================================================================\n"
