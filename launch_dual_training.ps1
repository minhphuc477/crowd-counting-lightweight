# ==============================================================================
# Dual Training Launcher for RMR ShanghaiTech Part A
# Runs both Step 1 (104k Baseline Recovery) and Step 2 (152k Next-Gen) in parallel
# VRAM-safe: Each uses batch_size=4 + grad_accum=2 (~1.1 GB VRAM each, fitting in 4GB VRAM)
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " LAUNCHING DUAL TRAINING RUNS (RMR CROWD COUNTING)" -ForegroundColor Cyan
Write-Host " Benchmark: ShanghaiTech Part A (Official Split 300 Train / 182 Test)" -ForegroundColor Yellow
Write-Host " 1. Run 1 (104k): sub60_v39_golden_recipe_wsd600.yaml" -ForegroundColor Green
Write-Host " 2. Run 2 (152k): sub60_v40_nextgen_200k_wsd600.yaml" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Cyan

$env:MALLOC_ARENA_MAX = "2"
$env:OMP_NUM_THREADS = "2"

$job1 = Start-Process -FilePath "python" -ArgumentList "-m rmr_v3.train -c configs/rmr_research/sub60_v39_golden_recipe_wsd600.yaml" -PassThru -NoNewWindow
Start-Sleep -Seconds 5
$job2 = Start-Process -FilePath "python" -ArgumentList "-m rmr_v3.train -c configs/rmr_research/sub60_v40_nextgen_200k_wsd600.yaml" -PassThru -NoNewWindow

Write-Host "Both jobs launched successfully!" -ForegroundColor Green
Write-Host "Job 1 PID: $($job1.Id) -> Output: runs/sha_a/sub60_v39_golden_wsd600" -ForegroundColor Green
Write-Host "Job 2 PID: $($job2.Id) -> Output: runs/sha_a/sub60_v40_nextgen_200k_wsd600" -ForegroundColor Magenta
Write-Host "Monitor with: Get-Content runs/sha_a/sub60_v39_golden_wsd600/train_log.csv -Tail 10" -ForegroundColor Gray
Write-Host "Monitor with: Get-Content runs/sha_a/sub60_v40_nextgen_200k_wsd600/train_log.csv -Tail 10" -ForegroundColor Gray
