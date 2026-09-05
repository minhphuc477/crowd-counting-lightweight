$ErrorActionPreference = "Stop"

$models = @(
    @{ Name = "B0 (direct)"; Config = "configs/rmr/direct.yaml"; OutDir = "runs/sha_a/direct_seed42" },
    @{ Name = "B1 (region_loss)"; Config = "configs/rmr/region_loss.yaml"; OutDir = "runs/sha_a/region_loss_seed42" },
    @{ Name = "B2 (region_aux)"; Config = "configs/rmr/region_aux.yaml"; OutDir = "runs/sha_a/region_aux_seed42" },
    @{ Name = "B3a (local_refine)"; Config = "configs/rmr/local_refine.yaml"; OutDir = "runs/sha_a/local_refine_seed42" },
    @{ Name = "B3b (learned_project)"; Config = "configs/rmr/learned_project.yaml"; OutDir = "runs/sha_a/learned_project_seed42" },
    @{ Name = "B5-P (rmr_projected_t2)"; Config = "configs/rmr/rmr_projected_t2.yaml"; OutDir = "runs/sha_a/rmr_projected_t2_seed42" }
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
        --eval-every 10
        
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished Model: $($m.Name)"
}

Write-Host "========================================================="
Write-Host "STAGE C FULL MATCHED MATRIX COMPLETED SUCCESSFULLY!"
Write-Host "========================================================="
