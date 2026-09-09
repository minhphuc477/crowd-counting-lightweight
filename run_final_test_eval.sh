#!/usr/bin/env bash
set -euo pipefail

# Guard against accidental unfreezing of the test set:
if [ "${RMR_ALLOW_TEST_EVAL:-0}" != "1" ]; then
    echo "ERROR: [DATA INTEGRITY GUARD] Test set evaluation is strictly guarded!" >&2
    echo "The test set must remain frozen until all architecture, loss, configuration," >&2
    echo "and checkpoint selection rules are permanently frozen and committed." >&2
    echo "To proceed with release benchmarking, set the environment variable:" >&2
    echo "  export RMR_ALLOW_TEST_EVAL=1" >&2
    echo "and re-run this script." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/scripts/release/run_final_test_eval.sh"
