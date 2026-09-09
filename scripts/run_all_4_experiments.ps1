param (
    [switch]$Fresh = $false,
    [switch]$AllowDirty = $true,
    [switch]$AllowCrossCommitResume = $true
)

$ErrorActionPreference = "Stop"

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$currentCommit = (git rev-parse HEAD 2>$null)
if ($currentCommit) { $currentCommit = $currentCommit.Trim() } else { $currentCommit = "unknown" }

$models = @(
    @{ Name = "1/4: V3-A (Probabilistic Uniform Control)"; Config = "configs/rmr_v3/probabilistic_uniform.yaml"; OutDir = "runs/sha_a/rmr_v3_uniform_control_seed42"; Module = "rmr_v3.train"; EvalModule = "rmr_v3.eval" },
    @{ Name = "2/4: V3-B (Reliability Weighted - Proposed)"; Config = "configs/rmr_v3/reliability_weighted.yaml"; OutDir = "runs/sha_a/rmr_v3_rw_seed42"; Module = "rmr_v3.train"; EvalModule = "rmr_v3.eval" },
    @{ Name = "3/4: B5-P (Canonical Deterministic Baseline)"; Config = "configs/rmr_v2/rmr_projected_t2.yaml"; OutDir = "runs/sha_a/b5p_canonical_seed42"; Module = "rmr_v2.train"; EvalModule = "rmr_v2.eval" },
    @{ Name = "4/4: RMR-v4 Full Candidate"; Config = "configs/rmr_v4/final_candidate.yaml"; OutDir = "runs/sha_a/rmr_v4_candidate_seed42"; Module = "rmr_v3.train"; EvalModule = "rmr_v3.eval" }
)

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  RMR FULL 4-MODEL BENCHMARK SUITE (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "  Order: V3-A -> V3-B -> B5-P -> V4-Candidate" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

foreach ($m in $models) {
    $outDir = $m.OutDir
    $bestCkpt = "$outDir/best_val_mae.pt"
    $lastCkpt = "$outDir/last.pt"
    $evalDir = "$outDir/eval_test"

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting: $($m.Name)" -ForegroundColor Yellow
    Write-Host "  Config: $($m.Config) | OutDir: $outDir" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

    if (-not $Fresh -and (Test-Path "$evalDir/summary.json") -and (Test-Path $bestCkpt)) {
        Write-Host "  [SKIP] Found valid artifacts at $outDir. Skipping training." -ForegroundColor Green
    } else {
        if (-not $Fresh -and (Test-Path $lastCkpt)) {
            Write-Host "  [RESUME] Resuming from $lastCkpt..." -ForegroundColor Green
            $trainArgs = @("-m", $m.Module, "--config", $m.Config, "--resume", $lastCkpt, "--allow-cross-commit-resume")
        } else {
            $trainArgs = @("-m", $m.Module, "--config", $m.Config, "--overwrite")
        }
        & $pythonExe $trainArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training failed for $($m.Name) with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }

    if (-not (Test-Path $bestCkpt)) {
        Write-Error "Best checkpoint not found at $bestCkpt"
        exit 1
    }

    Write-Host "  [EVAL] Evaluating $bestCkpt on data/sha_a_test.jsonl..." -ForegroundColor Cyan
    & $pythonExe -m $m.EvalModule --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Evaluation failed for $($m.Name)"
        exit $LASTEXITCODE
    }
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished: $($m.Name)`n" -ForegroundColor Green
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ALL 4 RUNS COMPLETED! RUNNING PAIRED COMPARISONS..." -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

$predV3A = "runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
$predV3B = "runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
$predB5P = "runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"
$predV4 = "runs/sha_a/rmr_v4_candidate_seed42/eval_test/predictions.csv"

if ((Test-Path $predV3A) -and (Test-Path $predV3B)) {
    & $pythonExe -m rmr_count.aggregate --compare $predV3A $predV3B --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_v3a_vs_v3b.json"
}

if ((Test-Path $predB5P) -and (Test-Path $predV3B)) {
    & $pythonExe -m rmr_count.aggregate --compare $predB5P $predV3B --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3b.json"
}

if ((Test-Path $predV3B) -and (Test-Path $predV4)) {
    & $pythonExe -m rmr_count.aggregate --compare $predV3B $predV4 --name-a "V3-B_RW" --name-b "V4_Candidate" --pred-col pred --output "runs/sha_a/comparison_v3b_vs_v4_candidate.json"
}

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  COMPREHENSIVE 4-MODEL SUITE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
