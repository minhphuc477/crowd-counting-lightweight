#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RMR-v4 PARALLEL ABLATION EXPERIMENT RUNNER (16GB GPU Mode)
# Runs the 4 decisive RMR-v4 ablation models simultaneously in parallel:
#   1. V4-S:  Mean/Std Region Stats Only       (configs/rmr_v4/mean_std_regions.yaml)
#   2. V4-N:  Native Scale Pooling Only        (configs/rmr_v4/native_pooling.yaml)
#   3. V4-NS: Native Pooling + Mean/Std Combo   (configs/rmr_v4/native_meanstd.yaml)
#   4. V4-DM: Multi-Scale DM Loss Only         (configs/rmr_v4/multiscale_dm.yaml)
# ==============================================================================

# Optimize CUDA allocator for multi-process concurrency on 16GB GPU
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

LOG_DIR="runs/sha_a/parallel_logs"
mkdir -p "$LOG_DIR"

echo -e "\n================================================================================"
echo "  RMR-V4 DECISIVE 4-MODEL ABLATION SUITE (PARALLEL 16GB VRAM MODE)"
echo "  Hardware Target: Ubuntu / 16GB VRAM GPU"
echo "  Launching 4 models concurrently:"
echo "    [1] V4-S  (Mean/Std region stats only)"
echo "    [2] V4-N  (Native Scale Pooling P4/P8/P16 only)"
echo "    [3] V4-NS (Native Pooling + Mean/Std combo)"
echo "    [4] V4-DM (Multi-Scale DM loss only)"
echo "  Logs saved to: $LOG_DIR/"
echo -e "================================================================================\n"

# Launch helper
launch_ablation() {
    local tag="$1"
    local name="$2"
    local cfg="$3"
    local out_dir="$4"
    local log_file="$LOG_DIR/${tag}.log"

    mkdir -p "$out_dir"
    local last_ckpt="$out_dir/last.pt"
    local train_args=("-m" "rmr_v3.train" "--config" "$cfg")

    if [ -f "$last_ckpt" ]; then
        echo "  [$tag: RESUME] Found checkpoint at $last_ckpt. Resuming..."
        train_args+=("--resume" "$last_ckpt" "--allow-cross-commit-resume")
    else
        train_args+=("--overwrite")
    fi

    echo "  [LAUNCH] $name -> Logging to $log_file"
    "$PYTHON_EXE" "${train_args[@]}" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting all 4 ablation trainings concurrently..."

PID_V4S=$(launch_ablation "v4_s"  "V4-S  (Mean/Std Stats)"       "configs/rmr_v4/mean_std_regions.yaml" "runs/sha_a/rmr_v4_mean_std_seed42")
PID_V4N=$(launch_ablation "v4_n"  "V4-N  (Native Pooling)"       "configs/rmr_v4/native_pooling.yaml"   "runs/sha_a/rmr_v4_native_pooling_seed42")
PID_V4NS=$(launch_ablation "v4_ns" "V4-NS (Native + Mean/Std)"    "configs/rmr_v4/native_meanstd.yaml"   "runs/sha_a/rmr_v4_native_meanstd_seed42")
PID_V4DM=$(launch_ablation "v4_dm" "V4-DM (Multi-Scale DM)"       "configs/rmr_v4/multiscale_dm.yaml"    "runs/sha_a/rmr_v4_multiscale_dm_seed42")

echo -e "\nAll 4 processes are running in parallel on GPU:"
echo "  [1] PID $PID_V4S  | V4-S  | Live log: tail -f $LOG_DIR/v4_s.log"
echo "  [2] PID $PID_V4N  | V4-N  | Live log: tail -f $LOG_DIR/v4_n.log"
echo "  [3] PID $PID_V4NS | V4-NS | Live log: tail -f $LOG_DIR/v4_ns.log"
echo "  [4] PID $PID_V4DM | V4-DM | Live log: tail -f $LOG_DIR/v4_dm.log"

echo -e "\nMonitor GPU memory in a separate terminal: watch -n 1 nvidia-smi"
echo "Waiting for all 4 ablation trainings to complete..."

# Wait for all processes
wait $PID_V4S  || echo "Warning: V4-S exited with code $?"
wait $PID_V4N  || echo "Warning: V4-N exited with code $?"
wait $PID_V4NS || echo "Warning: V4-NS exited with code $?"
wait $PID_V4DM || echo "Warning: V4-DM exited with code $?"

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] ALL 4 TRAININGS COMPLETED!"
echo "================================================================================"
echo "  EVALUATING ALL 4 MODELS ON SHANGHAITECH PART A TEST SET (182 IMAGES)..."
echo -e "================================================================================\n"

eval_model() {
    local tag="$1"
    local name="$2"
    local out_dir="$3"
    local best_ckpt="$out_dir/best_val_mae.pt"
    local eval_dir="$out_dir/eval_test"

    echo "Evaluating $name ($best_ckpt)..."
    if [ ! -f "$best_ckpt" ]; then
        echo "ERROR: Checkpoint not found: $best_ckpt" >&2
        return 1
    fi

    "$PYTHON_EXE" -m rmr_v3.eval \
        --checkpoint "$best_ckpt" \
        --manifest "data/sha_a_test.jsonl" \
        --output-dir "$eval_dir"
}

eval_model "v4_s"  "V4-S (Mean/Std Stats)"       "runs/sha_a/rmr_v4_mean_std_seed42"
eval_model "v4_n"  "V4-N (Native Pooling)"       "runs/sha_a/rmr_v4_native_pooling_seed42"
eval_model "v4_ns" "V4-NS (Native + Mean/Std)"    "runs/sha_a/rmr_v4_native_meanstd_seed42"
eval_model "v4_dm" "V4-DM (Multi-Scale DM)"       "runs/sha_a/rmr_v4_multiscale_dm_seed42"

echo -e "\n================================================================================"
echo "  RUNNING PAIRED STATISTICAL COMPARISONS (T-TEST, WILCOXON, BOOTSTRAP CI)..."
echo -e "================================================================================\n"

PRED_V3B="runs/sha_a/historical_commit_91c0b841/rmr_v3_rw_seed42/eval_test/predictions.csv"
PRED_V4S="runs/sha_a/rmr_v4_mean_std_seed42/eval_test/predictions.csv"
PRED_V4N="runs/sha_a/rmr_v4_native_pooling_seed42/eval_test/predictions.csv"
PRED_V4NS="runs/sha_a/rmr_v4_native_meanstd_seed42/eval_test/predictions.csv"
PRED_V4DM="runs/sha_a/rmr_v4_multiscale_dm_seed42/eval_test/predictions.csv"

# 1. Baseline V3-B vs V4-S
if [ -f "$PRED_V3B" ] && [ -f "$PRED_V4S" ]; then
    echo "--- [1/6] Paired Comparison: V3-B (RW-RMR) vs V4-S (Mean+Std) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4S" \
        --name-a "V3-B_RW" --name-b "V4-S_MeanStd" --pred-col pred \
        --output "runs/sha_a/comparison_v3b_vs_v4_mean_std.json"
fi

# 2. Baseline V3-B vs V4-N
if [ -f "$PRED_V3B" ] && [ -f "$PRED_V4N" ]; then
    echo -e "\n--- [2/6] Paired Comparison: V3-B (RW-RMR) vs V4-N (Native Pooling) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4N" \
        --name-a "V3-B_RW" --name-b "V4-N_NativePool" --pred-col pred \
        --output "runs/sha_a/comparison_v3b_vs_v4_native_pooling.json"
fi

# 3. Baseline V3-B vs V4-NS
if [ -f "$PRED_V3B" ] && [ -f "$PRED_V4NS" ]; then
    echo -e "\n--- [3/6] Paired Comparison: V3-B (RW-RMR) vs V4-NS (Native + Mean/Std) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4NS" \
        --name-a "V3-B_RW" --name-b "V4-NS_Combo" --pred-col pred \
        --output "runs/sha_a/comparison_v3b_vs_v4_native_meanstd.json"
fi

# 4. Baseline V3-B vs V4-DM
if [ -f "$PRED_V3B" ] && [ -f "$PRED_V4DM" ]; then
    echo -e "\n--- [4/6] Paired Comparison: V3-B (RW-RMR) vs V4-DM (Multi-Scale DM) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V3B" "$PRED_V4DM" \
        --name-a "V3-B_RW" --name-b "V4-DM_Multiscale" --pred-col pred \
        --output "runs/sha_a/comparison_v3b_vs_v4_multiscale_dm.json"
fi

# 5. V4-S vs V4-NS (Causal contribution of adding Native Pooling to Mean/Std)
if [ -f "$PRED_V4S" ] && [ -f "$PRED_V4NS" ]; then
    echo -e "\n--- [5/6] Paired Comparison: V4-S (Mean/Std) vs V4-NS (Combo) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V4S" "$PRED_V4NS" \
        --name-a "V4-S_MeanStd" --name-b "V4-NS_Combo" --pred-col pred \
        --output "runs/sha_a/comparison_v4s_vs_v4ns.json"
fi

# 6. V4-N vs V4-NS (Causal contribution of adding Mean/Std to Native Pooling)
if [ -f "$PRED_V4N" ] && [ -f "$PRED_V4NS" ]; then
    echo -e "\n--- [6/6] Paired Comparison: V4-N (Native Pooling) vs V4-NS (Combo) ---"
    "$PYTHON_EXE" -m rmr_count.aggregate --compare "$PRED_V4N" "$PRED_V4NS" \
        --name-a "V4-N_NativePool" --name-b "V4-NS_Combo" --pred-col pred \
        --output "runs/sha_a/comparison_v4n_vs_v4ns.json"
fi

echo -e "\n================================================================================"
echo "  SUMMARY OF EVALUATION RESULTS (ShanghaiTech Part A - 182 Test Images)"
echo "================================================================================"
"$PYTHON_EXE" -c "
import json, os

models = [
    ('V4-S  (Mean/Std)', 'runs/sha_a/rmr_v4_mean_std_seed42/eval_test/summary.json'),
    ('V4-N  (Native Pool)', 'runs/sha_a/rmr_v4_native_pooling_seed42/eval_test/summary.json'),
    ('V4-NS (Combo)', 'runs/sha_a/rmr_v4_native_meanstd_seed42/eval_test/summary.json'),
    ('V4-DM (Multi-Scale DM)', 'runs/sha_a/rmr_v4_multiscale_dm_seed42/eval_test/summary.json'),
]

print(f'{'Model Name':<25} | {'Test MAE':<10} | {'Test RMSE':<10} | {'NAE':<8} | {'Bias':<8}')
print('-' * 70)
for name, p in models:
    if os.path.exists(p):
        d = json.load(open(p))
        mae = f'{d.get(\"MAE\", d.get(\"mae\", 0.0)):.2f}'
        rmse = f'{d.get(\"RMSE\", d.get(\"rmse\", 0.0)):.2f}'
        nae = f'{d.get(\"NAE\", d.get(\"nae\", 0.0)):.4f}'
        bias = f'{d.get(\"Bias\", d.get(\"bias\", 0.0)):+.2f}'
        print(f'{name:<25} | {mae:<10} | {rmse:<10} | {nae:<8} | {bias:<8}')
    else:
        print(f'{name:<25} | [Incomplete / Not Found]')
print('=' * 70)
"

echo -e "\nALL 4 RMR-V4 ABLATIONS HAVE COMPLETED SUCCESSFULLY!"
