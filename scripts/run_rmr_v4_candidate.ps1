param (
    [switch]$Fresh = $false,
    [switch]$AllowDirty = $false
)

$ErrorActionPreference = "Stop"

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$currentCommit = (git rev-parse HEAD).Trim()
$gitStatusRaw = git status --porcelain
$gitStatus = if ($null -ne $gitStatusRaw) { ("$gitStatusRaw").Trim() } else { "" }
$isDirty = [bool]$gitStatus

if ($isDirty -and (-not $AllowDirty)) {
    Write-Error "Working tree is dirty at git commit $currentCommit. Please commit or stash changes, or pass -AllowDirty."
    exit 1
}

$cfg = "configs/rmr_v4/final_candidate.yaml"
$outDir = "runs/sha_a/rmr_v4_candidate_seed42"
$bestCkpt = "$outDir/best_val_mae.pt"
$evalDir = "$outDir/eval_test"
$summaryJson = "$evalDir/summary.json"

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  RMR-V4 FULL CANDIDATE EXPERIMENT RUNNER" -ForegroundColor Cyan
Write-Host "  Features: Native Scale Pooling + Mean/Std Regional Stats + Multi-Scale DM Loss" -ForegroundColor Cyan
Write-Host "  Git HEAD: $currentCommit (Dirty: $isDirty)" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

# 1. Training Step
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Training: Full V4 Candidate" -ForegroundColor Yellow
Write-Host "  Config: $cfg | OutDir: $outDir" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

$lastCkpt = "$outDir/last.pt"
if (-not $Fresh -and (Test-Path $lastCkpt)) {
    Write-Host "  [RESUME] Found valid resume checkpoint at $lastCkpt. Resuming training from last epoch..." -ForegroundColor Green
    $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--resume", $lastCkpt, "--allow-cross-commit-resume")
} else {
    $trainArgs = @("-m", "rmr_v3.train", "--config", $cfg, "--overwrite")
}
Write-Host "  [TRAIN] Executing: $pythonExe $($trainArgs -join ' ')"
& $pythonExe $trainArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed for RMR-v4 Candidate with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 2. Evaluation Step
Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Evaluating on ShanghaiTech Part A Test Set (182 images)..." -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

if (-not (Test-Path $bestCkpt)) {
    Write-Error "Best checkpoint not found at $bestCkpt"
    exit 1
}

& $pythonExe -m rmr_v3.eval --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# 3. Paired Statistical Comparison against V3-B
$predV3B = "runs/sha_a/historical_commit_91c0b841/rmr_v3_rw_seed42/eval_test/predictions.csv"
$predV4 = "$evalDir/predictions.csv"
$compJson = "runs/sha_a/comparison_v3b_vs_v4_candidate.json"

if (Test-Path $predV3B) {
    Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running Paired Statistical Comparison: V3-B vs V4 Candidate..." -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
    & $pythonExe -m rmr_count.aggregate --compare $predV3B $predV4 --name-a "V3-B_RW" --name-b "V4_Candidate" --pred-col pred --output $compJson
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Paired comparison failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  RMR-V4 CANDIDATE EXPERIMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
