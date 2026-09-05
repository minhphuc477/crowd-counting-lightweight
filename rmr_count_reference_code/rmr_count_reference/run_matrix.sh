#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Pilot LR sweep on validation only. Freeze the selected LR before multi-seed final runs.
for lr in 1e-4 3e-4 1e-3; do
  python -m rmr_count.train \
    --config configs/rmr_t2.yaml \
    --seed 42 \
    --lr "$lr" \
    --output-dir "runs/sha_a/lr_sweep_rmr_t2_${lr}"
done

# Matched RQ matrix after choosing LR using validation only.
LR=3e-4   # replace only with the validation-selected value
for seed in 42 123 3407; do
  for cfg in direct region_loss region_aux learned_project rmr_t1 rmr_t2; do
    python -m rmr_count.train \
      --config "configs/${cfg}.yaml" \
      --seed "$seed" \
      --lr "$LR" \
      --output-dir "runs/sha_a/${cfg}_seed${seed}"
  done
done
