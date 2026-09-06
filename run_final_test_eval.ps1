$ErrorActionPreference = "Stop"

# Guard against accidental unfreezing of the test set:
if ($env:RMR_ALLOW_TEST_EVAL -ne "1") {
    Write-Error @"
[DATA INTEGRITY GUARD] Test set evaluation is strictly guarded!
The test set must remain frozen until all architecture, loss, configuration,
and checkpoint selection rules are permanently frozen and committed.
To proceed with release benchmarking, set the environment variable:
  `$env:RMR_ALLOW_TEST_EVAL = "1"
and re-run this script.
"@
    exit 1
}

& "$PSScriptRoot\scripts\release\run_final_test_eval.ps1"