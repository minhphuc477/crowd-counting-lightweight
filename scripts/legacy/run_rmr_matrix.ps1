$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$((Get-Location).Path);$($env:PYTHONPATH)"

# Matched RQ matrix after choosing LR using validation only.
# Pilot sweep indicates 1e-4 produces the most stable validation performance.
$LR = "1e-4"
$seeds = @(42, 123, 3407)
$cfgs = @("direct", "region_loss", "region_aux", "local_refine", "learned_project", "rmr_t1", "rmr_t2")

foreach ($seed in $seeds) {
    foreach ($cfg in $cfgs) {
        Write-Host "========================================================="
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting $cfg (seed $seed, lr $LR)"
        Write-Host "========================================================="
        
        .venv\Scripts\python -m rmr_count.train `
            --config "configs/rmr/$cfg.yaml" `
            --seed $seed `
            --lr $LR `
            --output-dir "runs/sha_a/${cfg}_seed${seed}"
            
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished $cfg (seed $seed)"
    }
}
Write-Host "Full RMR experimental matrix completed successfully!"
