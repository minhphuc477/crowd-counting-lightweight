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

$models = @(
    @{ Name = "B0 (direct)"; OutDir = "runs/sha_a/stage_c_b0_direct_seed42" },
    @{ Name = "B1 (region_loss)"; OutDir = "runs/sha_a/stage_c_b1_region_loss_seed42" },
    @{ Name = "B2 (region_aux)"; OutDir = "runs/sha_a/stage_c_b2_region_aux_seed42" },
    @{ Name = "B3a (local_refine)"; OutDir = "runs/sha_a/stage_c_b3a_local_refine_seed42" },
    @{ Name = "B3b (learned_project)"; OutDir = "runs/sha_a/stage_c_b3b_learned_project_seed42" },
    @{ Name = "B5-P (rmr_projected_t2)"; OutDir = "runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42" }
)

Write-Host "========================================================="
Write-Host "STAGE C FINAL POST-FREEZE TEST SET BENCHMARK"
Write-Host "========================================================="

foreach ($m in $models) {
    $ckptPath = "$($m.OutDir)/best_val_mae.pt"
    if (-not (Test-Path $ckptPath)) {
        $ckptPath = "$($m.OutDir)/last.pt"
    }

    if (-not (Test-Path $ckptPath)) {
        Write-Warning "Checkpoint not found for $($m.Name): $ckptPath. Skipping."
        continue
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Evaluating $($m.Name) on Test set using $ckptPath..."
    .venv\Scripts\python -m rmr_count.eval `
        --checkpoint $ckptPath `
        --manifest "data/sha_a_test.jsonl" `
        --out-dir "$($m.OutDir)/eval_test"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Completed test eval for: $($m.Name)"
}

Write-Host "========================================================="
Write-Host "STAGE C FINAL TEST EVALUATION COMPLETED!"
Write-Host "========================================================="
