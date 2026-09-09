#!/usr/bin/env bash
set -euo pipefail

# Guard against accidental unfreezing of the test set:
if [ "${RMR_ALLOW_TEST_EVAL:-0}" != "1" ]; then
    echo "ERROR: [DATA INTEGRITY GUARD] Test set evaluation is strictly guarded!" >&2
    echo "To proceed with release benchmarking, set the environment variable: export RMR_ALLOW_TEST_EVAL=1" >&2
    exit 1
fi

PYTHON_EXE=".venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python3"
fi

models=(
    "B0 (direct)|runs/sha_a/stage_c_b0_direct_seed42"
    "B1 (region_loss)|runs/sha_a/stage_c_b1_region_loss_seed42"
    "B2 (region_aux)|runs/sha_a/stage_c_b2_region_aux_seed42"
    "B3a (local_refine)|runs/sha_a/stage_c_b3a_local_refine_seed42"
    "B3b (learned_project)|runs/sha_a/stage_c_b3b_learned_project_seed42"
    "B5-P (rmr_projected_t2)|runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42"
)

echo "========================================================="
echo "STAGE C FINAL POST-FREEZE TEST SET BENCHMARK (Linux)"
echo "========================================================="

for item in "${models[@]}"; do
    IFS="|" read -r name out_dir <<< "$item"
    ckpt_path="$out_dir/best_val_mae.pt"
    if [ ! -f "$ckpt_path" ]; then
        ckpt_path="$out_dir/last.pt"
    fi

    if [ ! -f "$ckpt_path" ]; then
        echo "WARNING: Checkpoint not found for $name: $ckpt_path. Skipping."
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Evaluating $name on Test set using $ckpt_path..."
    "$PYTHON_EXE" -m rmr_count.eval \
        --checkpoint "$ckpt_path" \
        --manifest "data/sha_a_test.jsonl" \
        --out-dir "$out_dir/eval_test"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed test eval for: $name"
done

echo "========================================================="
echo "STAGE C FINAL TEST EVALUATION COMPLETED!"
echo "========================================================="
