$ErrorActionPreference = "Stop"

Write-Host "========================================================="
Write-Host "Starting Stage B Pilot: B2 (region_aux)"
Write-Host "========================================================="

.venv\Scripts\python -m rmr_count.train `
    --config configs/rmr/region_aux.yaml `
    --seed 42 `
    --lr 0.0001 `
    --output-dir runs/sha_a/pilot_b2_seed42 `
    --epochs 100 `
    --eval-every 5

Write-Host "========================================================="
Write-Host "Starting Stage B Pilot: B3b (learned_project)"
Write-Host "========================================================="

.venv\Scripts\python -m rmr_count.train `
    --config configs/rmr/learned_project.yaml `
    --seed 42 `
    --lr 0.0001 `
    --output-dir runs/sha_a/pilot_b3b_seed42 `
    --epochs 100 `
    --eval-every 5

Write-Host "========================================================="
Write-Host "Stage B Pilot Completed!"
Write-Host "========================================================="
