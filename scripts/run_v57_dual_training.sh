#!/bin/bash
# ==============================================================================
# Dual Training Launcher for RMR V57 on Ubuntu Linux (RTX 5070 Ti 16GB)
# Runs Wave 1 Proposed models (Track A + Track B) in parallel:
#   - Track A (103,959 params): ~2.1 GB VRAM
#   - Track B (144,960 params): ~3.4 GB VRAM
# Total VRAM footprint: ~5.5 GB (fits easily within 16GB GDDR7)
# ==============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="${REPO_ROOT}/logs/rmr_v57"
mkdir -p "${LOG_DIR}"

echo "=========================================================="
echo " LAUNCHING RMR V57 PROPOSED DUAL EXPERIMENTS"
echo " Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Branch: RMR | Commit: $(git rev-parse --short HEAD)"
echo " 1. Track A: v57_track_a_proposed_multistep_rn.yaml (103,959 params)"
echo " 2. Track B: v57_track_b_proposed_deep_texture_rn.yaml (144,960 params)"
echo "=========================================================="

# Launch Track A in background
python train.py --config configs/rmr_v57/v57_track_a_proposed_multistep_rn.yaml \
    > "${LOG_DIR}/train_v57_track_a.log" 2>&1 &
PID_TRACK_A=$!
echo "[STARTED] Track A (PID: ${PID_TRACK_A}) -> logs/rmr_v57/train_v57_track_a.log"

sleep 3

# Launch Track B in background
python train.py --config configs/rmr_v57/v57_track_b_proposed_deep_texture_rn.yaml \
    > "${LOG_DIR}/train_v57_track_b.log" 2>&1 &
PID_TRACK_B=$!
echo "[STARTED] Track B (PID: ${PID_TRACK_B}) -> logs/rmr_v57/train_v57_track_b.log"

echo ""
echo "Both experiments are running in the background."
echo "Monitor Track A: tail -f ${LOG_DIR}/train_v57_track_a.log"
echo "Monitor Track B: tail -f ${LOG_DIR}/train_v57_track_b.log"
echo "Monitor CSV metrics:"
echo "  tail -n 10 runs/sha_a/rmr_v57/v57_track_a_proposed_multistep_rn/train_log.csv"
echo "  tail -n 10 runs/sha_a/rmr_v57/v57_track_b_proposed_deep_texture_rn/train_log.csv"
