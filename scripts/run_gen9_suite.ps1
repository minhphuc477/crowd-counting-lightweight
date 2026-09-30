# PowerShell Runner for Gen 9 Mathematical Fixes & Breakthrough Suite
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "4" }

$models = @(
    # 1. Bug Fix 1: curvature_alpha_init = 0.0 (active from epoch 1)
    @{ RunId = "g9_fix_curvature_init";    Config = "configs/rmr_research/gen9/g9_fix_curvature_init.yaml" },

    # 2. Bug Fix 2: hurdle_gating_mode = occupancy (eliminates 15-25% dense mass erosion)
    @{ RunId = "g9_fix_hurdle_occupancy";   Config = "configs/rmr_research/gen9/g9_fix_hurdle_occupancy.yaml" },

    # 3. Asymmetric Morozov Discrepancy (gamma_under = 0.35 bias correction)
    @{ RunId = "g9_asym_morozov_e5";        Config = "configs/rmr_research/gen9/g9_asym_morozov_e5.yaml" },

    # 4. Softer MCP Firm Threshold (proximal_mu = 1.5)
    @{ RunId = "g9_softer_firm_mu15";       Config = "configs/rmr_research/gen9/g9_softer_firm_mu15.yaml" },

    # 5. DCT-II Neumann Spectral Loss on corrected baseline
    @{ RunId = "g9_dct_spectral_fixed";     Config = "configs/rmr_research/gen9/g9_dct_spectral_fixed.yaml" },

    # 6. Compound Best: All 4 orthogonal fixes synthesized
    @{ RunId = "g9_compound_all_fixes";     Config = "configs/rmr_research/gen9/g9_compound_all_fixes.yaml" }
)

$logDir = "runs/sha_a/suite_logs_gen9"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR GEN 9 COMPREHENSIVE EXPERIMENT SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "  6 Experiments | Target: Sub-60 MAE (< 60.0)" -ForegroundColor Cyan
Write-Host "  Baseline: sub60_e5 (71.51 MAE) | Constraints: <= 105k params, Zero KD" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

foreach ($m in $models) {
    $runId = $m.RunId
    $cfg = $m.Config
    $log = "$logDir/$runId.log"

    Write-Host "`n>>> Starting training for: $runId ($cfg)" -ForegroundColor Green
    & $py -m rmr_v3.train --config $cfg --run-id $runId --overwrite 2>&1 | Tee-Object -FilePath $log
    Write-Host ">>> Completed training: $runId" -ForegroundColor Yellow

    $ckptPath = "runs/sha_a/$runId/best_val_mae.pt"
    if (Test-Path $ckptPath) {
        Write-Host ">>> Evaluating Standard Test: $ckptPath" -ForegroundColor Magenta
        & $py -m rmr_v3.evaluate `
            --checkpoint $ckptPath `
            --manifest data/sha_a_test.jsonl `
            --crop-size 512 `
            --tta $false `
            --output-json "runs/sha_a/$runId/eval_test_canonical.json"
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "  GEN 9 EXPERIMENT SUITE COMPLETED!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
