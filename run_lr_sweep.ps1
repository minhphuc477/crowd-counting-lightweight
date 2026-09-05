$ErrorActionPreference = "Stop"
$lrs = @("1e-4", "3e-4", "1e-3")

foreach ($lr in $lrs) {
    Write-Host "========================================================="
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting LR Sweep: lr = $lr"
    Write-Host "========================================================="
    
    $outDir = "runs/sha_a/lr_sweep_fixed_rmr_t2_$lr"
    
    .venv\Scripts\python -m rmr_count.train `
        --config configs/rmr/rmr_t2.yaml `
        --seed 42 `
        --lr $lr `
        --output-dir $outDir `
        --epochs 100 `
        --eval-every 5
        
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished LR Sweep: lr = $lr"
}
Write-Host "All LR sweeps completed successfully!"
