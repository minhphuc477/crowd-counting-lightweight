# Canonical RMR & RMR-v3 Benchmark Suite Runner
# Strictly adheres to:
# 1. Zero ad-hoc splits (300 train_all, 182 test).
# 2. Sequential execution for maximum CUDA throughput and thermal stability.
# 3. Direct full-image MAE/RMSE headline metrics.
# 4. Rigorous paired comparison (t-test, Wilcoxon, bootstrap 95% CI).

$ErrorActionPreference = "Stop"

$pythonExe = ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$runs = @(
    @{
        Name = "V3-A (Probabilistic Uniform Control)"
        Module = "rmr_v3.train"
        Config = "configs/rmr_v3/probabilistic_uniform.yaml"
        OutDir = "runs/sha_a/rmr_v3_uniform_control_seed42"
        EvalModule = "rmr_v3.eval"
    },
    @{
        Name = "V3-B (Reliability Weighted - Proposed)"
        Module = "rmr_v3.train"
        Config = "configs/rmr_v3/reliability_weighted.yaml"
        OutDir = "runs/sha_a/rmr_v3_rw_seed42"
        EvalModule = "rmr_v3.eval"
    },
    @{
        Name = "B5-P (Canonical Deterministic Baseline)"
        Module = "rmr_v2.train"
        Config = "configs/rmr_v2/rmr_projected_t2.yaml"
        OutDir = "runs/sha_a/b5p_canonical_seed42"
        EvalModule = "rmr_v2.eval"
    }
)

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  CANONICAL RMR / RMR-V3 BENCHMARK SUITE" -ForegroundColor Cyan
Write-Host "  Order: V3-A -> V3-B -> B5-P (Sequential execution on GPU)" -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

foreach ($run in $runs) {
    $name = $run.Name
    $cfg = $run.Config
    $outDir = $run.OutDir
    $trainMod = $run.Module
    $evalMod = $run.EvalModule
    $bestCkpt = "$outDir/best_val_mae.pt"
    $evalDir = "$outDir/eval_test"
    $summaryJson = "$evalDir/summary.json"

    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting: $name" -ForegroundColor Yellow
    Write-Host "  Config: $cfg | OutDir: $outDir" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Yellow

    # Training step: check if already completed with valid artifacts
    $artifactsValid = $false
    if ((Test-Path $summaryJson) -and (Test-Path $bestCkpt)) {
        try {
            $sumContent = Get-Content $summaryJson -Raw | ConvertFrom-Json
            if ($sumContent -and (Get-Item $bestCkpt).Length -gt 1000) {
                $artifactsValid = $true
            }
        } catch {
            $artifactsValid = $false
        }
    }

    if ($artifactsValid) {
        Write-Host "  [SKIP] Verified valid evaluation summary and checkpoint at $outDir. Skipping training." -ForegroundColor Green
    } else {
        $trainArgs = @("-m", $trainMod, "--config", $cfg)
        $lastCkpt = "$outDir/last.pt"
        if (Test-Path $lastCkpt) {
            Write-Host "  [RESUME] Found existing last.pt at $lastCkpt, resuming..." -ForegroundColor Magenta
            $trainArgs += @("--resume", $lastCkpt)
        }

        Write-Host "  [TRAIN] Executing: $pythonExe $($trainArgs -join ' ')"
        & $pythonExe $trainArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Training failed for $name with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }

    # Evaluation step: evaluate best_val_mae.pt on test set
    $evalValid = $false
    if (Test-Path $summaryJson) {
        try {
            $sumContent = Get-Content $summaryJson -Raw | ConvertFrom-Json
            if ($sumContent) { $evalValid = $true }
        } catch {
            $evalValid = $false
        }
    }
    if (-not $evalValid) {
        if (-not (Test-Path $bestCkpt)) {
            Write-Error "Checkpoint not found: $bestCkpt"
            exit 1
        }
        Write-Host "  [EVAL] Evaluating $bestCkpt on data/sha_a_test.jsonl..." -ForegroundColor Cyan
        & $pythonExe -m $evalMod --checkpoint $bestCkpt --manifest "data/sha_a_test.jsonl" --output-dir $evalDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Evaluation failed for $name with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finished: $name`n" -ForegroundColor Green
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ALL 3 RUNS COMPLETED. RUNNING PAIRED COMPARISONS AND AGGREGATION..." -ForegroundColor Cyan
Write-Host "================================================================================`n" -ForegroundColor Cyan

$predV3A = "runs/sha_a/rmr_v3_uniform_control_seed42/eval_test/predictions.csv"
$predV3B = "runs/sha_a/rmr_v3_rw_seed42/eval_test/predictions.csv"
$predB5P = "runs/sha_a/b5p_canonical_seed42/eval_test/predictions.csv"

# Comparison 1: V3-A vs V3-B (Causal test of reliability weighting W = I vs W = diag(w_R))
Write-Host "--- Comparison 1: V3-A (Uniform) vs V3-B (Reliability Weighted) [MAIN HYPOTHESIS] ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predV3A $predV3B --name-a "V3-A_Uniform" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_v3a_vs_v3b.json"

# Comparison 2: B5-P vs V3-A (Causal test of Probabilistic NB Regional Formulation)
Write-Host "`n--- Comparison 2: B5-P (Deterministic) vs V3-A (Probabilistic Uniform) ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predB5P $predV3A --name-a "B5-P_Deterministic" --name-b "V3-A_Uniform" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3a.json"

# Comparison 3: B5-P vs V3-B (Overall improvement from Baseline to Proposed)
Write-Host "`n--- Comparison 3: B5-P (Deterministic) vs V3-B (Reliability Weighted) ---" -ForegroundColor Yellow
& $pythonExe -m rmr_count.aggregate --compare $predB5P $predV3B --name-a "B5-P_Deterministic" --name-b "V3-B_RW" --pred-col pred --output "runs/sha_a/comparison_b5p_vs_v3b.json"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host "  CANONICAL SUITE EXECUTION AND COMPARISONS COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "================================================================================`n" -ForegroundColor Green
