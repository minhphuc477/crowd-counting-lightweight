$ErrorActionPreference = "Stop"
$lrs = @("1e-4", "3e-5", "3e-4")

foreach ($lr in $lrs) {
    Write-Host "========================================================="
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting RMR-P LR Screen: lr = $lr"
    Write-Host "========================================================="
    
    $outDir = "runs/sha_a/lr_screen_projected_$lr"
    
    .venv\Scripts\python -m rmr_count.train `
        --config configs/rmr/rmr_projected_t2.yaml `
        --seed 42 `
        --lr $lr `
        --output-dir $outDir `
        --epochs 100 `
        --eval-every 5
        
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished RMR-P LR Screen: lr = $lr"
}
Write-Host "All RMR-P LR screens completed successfully!"
