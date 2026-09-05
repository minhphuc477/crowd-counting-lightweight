$ErrorActionPreference = "Stop"

$models = @(
    @{ Name = "B0 (direct)"; Config = "configs/rmr/direct.yaml"; OutDir = "runs/sha_a/stage_c_b0_direct_seed42" },
    @{ Name = "B1 (region_loss)"; Config = "configs/rmr/region_loss.yaml"; OutDir = "runs/sha_a/stage_c_b1_region_loss_seed42" },
    @{ Name = "B2 (region_aux)"; Config = "configs/rmr/region_aux.yaml"; OutDir = "runs/sha_a/stage_c_b2_region_aux_seed42" },
    @{ Name = "B3a (local_refine)"; Config = "configs/rmr/local_refine.yaml"; OutDir = "runs/sha_a/stage_c_b3a_local_refine_seed42" },
    @{ Name = "B3b (learned_project)"; Config = "configs/rmr/learned_project.yaml"; OutDir = "runs/sha_a/stage_c_b3b_learned_project_seed42" },
    @{ Name = "B5-P (rmr_projected_t2)"; Config = "configs/rmr/rmr_projected_t2.yaml"; OutDir = "runs/sha_a/stage_c_b5_p_rmr_projected_t2_seed42" }
)

Write-Host "========================================================="
Write-Host "STARTING STAGE C FULL MATCHED MATRIX (1000 EPOCHS, SEED 42)"
Write-Host "========================================================="

foreach ($m in $models) {
    Write-Host "---------------------------------------------------------"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Model: $($m.Name)"
    Write-Host "Config: $($m.Config) | Output: $($m.OutDir)"
    Write-Host "---------------------------------------------------------"
    
    .venv\Scripts\python -m rmr_count.train `
        --config $m.Config `
        --seed 42 `
        --lr 0.0001 `
        --output-dir $m.OutDir `
        --epochs 1000 `
        --eval-every 10 `
        --patience 10

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training completed/stopped for: $($m.Name)"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Evaluating $($m.Name) on Val set..."
    .venv\Scripts\python -m rmr_count.eval `
        --checkpoint "$($m.OutDir)/best_val_mae.pt" `
        --manifest "data/sha_a_val.jsonl" `
        --out-dir "$($m.OutDir)/eval_val"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Evaluating $($m.Name) on Test set..."
    .venv\Scripts\python -m rmr_count.eval `
        --checkpoint "$($m.OutDir)/best_val_mae.pt" `
        --manifest "data/sha_a_test.jsonl" `
        --out-dir "$($m.OutDir)/eval_test"

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Completed all evals for: $($m.Name)"
}

Write-Host "========================================================="
Write-Host "STAGE C FULL MATCHED MATRIX COMPLETED SUCCESSFULLY!"
Write-Host "========================================================="
