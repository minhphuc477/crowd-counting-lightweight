#!/usr/bin/env bash
# ==============================================================================
# SUB-60 v37 EXPERIMENTAL SUITE RUNNER (ShanghaiTech Part A)
# Framework: Radon Measure Recovery (RMR) - Standalone (0.0% KD)
# Features:
# - Resource-bounded: MALLOC_ARENA_MAX=2, OMP_NUM_THREADS=2, workers=2 (<1.5GB RAM/run)
# - VRAM-bounded: ~1.1GB - 1.35GB per process (parallel runs safe on single GPU)
# Targets:
#   1. track_a       : Stride 4 (101k) - m04 Cured with Faithful Beta-NB + pad (Legacy Product Gate)
#   2. track_a_occ   : Stride 4 (101k) - m04 Cured with Mass-Preserving Occupancy Gate
#   3. track_b       : Stride 2 (174k) - FOV-64 Cured with dm_block_px=32 + Density-Adaptive Morozov
#   4. track_b_k20   : Stride 2 (174k) - FOV-64 Cured with Constant Kappa=20 Dirichlet
# Usage:
#   bash scripts/run_sub60_v37_suite.sh parallel      # Run all 4 in background simultaneously
#   bash scripts/run_sub60_v37_suite.sh sequential    # Run all 4 sequentially one by one
#   bash scripts/run_sub60_v37_suite.sh core          # Run only the 2 primary tracks (Track A & Track B)
#   bash scripts/run_sub60_v37_suite.sh <target_name> # Run an individual experiment
# ==============================================================================

set -euo pipefail

# Ensure memory bounds are exported globally
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=2

mkdir -p logs

declare -A CONFIGS=(
  ["track_a"]="configs/rmr_research/sub60_v37_track_a_stride4_m04_cured.yaml"
  ["track_a_occ"]="configs/rmr_research/sub60_v37_track_a_occupancy_101k.yaml"
  ["track_b"]="configs/rmr_research/sub60_v37_track_b_stride2_fov64_cured.yaml"
  ["track_b_k20"]="configs/rmr_research/sub60_v37_track_b_k20_morozov_174k.yaml"
)

run_single_bg() {
  local target="$1"
  local cfg="${CONFIGS[$target]}"
  local log_file="logs/sub60_${target}.log"
  echo "[LAUNCH] Starting ${target} in background -> log: ${log_file}"
  nohup python rmr_v3/train.py \
    -c "${cfg}" \
    --device cuda \
    --workers 2 \
    --num-threads 2 \
    > "${log_file}" 2>&1 &
  local pid=$!
  echo "         PID: ${pid} | Config: ${cfg}"
}

run_single_fg() {
  local target="$1"
  local cfg="${CONFIGS[$target]}"
  echo "================================================================================"
  echo "[RUNNING] Starting ${target} interactively -> Config: ${cfg}"
  echo "================================================================================"
  python rmr_v3/train.py \
    -c "${cfg}" \
    --device cuda \
    --workers 2 \
    --num-threads 2
}

MODE="${1:-core}"

case "${MODE}" in
  parallel|all)
    echo "=== Launching all 4 experiments in parallel (Background) ==="
    run_single_bg "track_a"
    run_single_bg "track_a_occ"
    run_single_bg "track_b"
    run_single_bg "track_b_k20"
    echo ""
    echo "All 4 experiments launched! Monitor progress with:"
    echo "  tail -f logs/sub60_*.log"
    ;;

  core)
    echo "=== Launching 2 primary core tracks in parallel (Track A & Track B) ==="
    run_single_bg "track_a"
    run_single_bg "track_b"
    echo ""
    echo "Track A and Track B launched! Monitor progress with:"
    echo "  tail -f logs/sub60_track_a.log logs/sub60_track_b.log"
    ;;

  sequential)
    echo "=== Running all 4 experiments sequentially ==="
    run_single_fg "track_a"
    run_single_fg "track_a_occ"
    run_single_fg "track_b"
    run_single_fg "track_b_k20"
    ;;

  track_a|track_a_occ|track_b|track_b_k20)
    run_single_fg "${MODE}"
    ;;

  *)
    echo "Unknown option: ${MODE}"
    echo "Valid options: parallel, core, sequential, track_a, track_a_occ, track_b, track_b_k20"
    exit 1
    ;;
esac
