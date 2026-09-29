# PowerShell Runner for Gen 7 Factorial Matrix & Breakthrough Ablation Suite
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Optional: Intra-op thread count to avoid CPU thread contention
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "4" }

$models = @(
    # 1. Base 4 Isolated Innovations
    @{ RunId = "g7_h4_drs";                 Config = "configs/rmr_research/gen7/g7_h4_drs.yaml" },
    @{ RunId = "g7_h1_fidt";                Config = "configs/rmr_research/gen7/g7_h1_fidt.yaml" },
    @{ RunId = "g7_h2_chfl";                Config = "configs/rmr_research/gen7/g7_h2_chfl.yaml" },
    @{ RunId = "g7_cdw_factorized_diag";    Config = "configs/rmr_research/gen7/g7_cdw_factorized_diag.yaml" },

    # 2. Pairwise Interactions (2-way Orthogonal Factorials)
    @{ RunId = "g7_h4_h1_drs_fidt";         Config = "configs/rmr_research/gen7/g7_h4_h1_drs_fidt.yaml" },
    @{ RunId = "g7_h4_h2_drs_chfl";         Config = "configs/rmr_research/gen7/g7_h4_h2_drs_chfl.yaml" },
    @{ RunId = "g7_h4_cdw";                 Config = "configs/rmr_research/gen7/g7_h4_cdw.yaml" },
    @{ RunId = "g7_h1_h2_fidt_chfl";        Config = "configs/rmr_research/gen7/g7_h1_h2_fidt_chfl.yaml" },
    @{ RunId = "g7_cdw_h1_fidt";            Config = "configs/rmr_research/gen7/g7_cdw_h1_fidt.yaml" },
    @{ RunId = "g7_cdw_h2_chfl";            Config = "configs/rmr_research/gen7/g7_cdw_h2_chfl.yaml" },

    # 3. Triads & Champion Synthesis
    @{ RunId = "g7_triad_drs_fidt_cdw";     Config = "configs/rmr_research/gen7/g7_triad_drs_fidt_cdw.yaml" },
    @{ RunId = "g7_triad_drs_chfl_cdw";     Config = "configs/rmr_research/gen7/g7_triad_drs_chfl_cdw.yaml" },
    @{ RunId = "g7_champion_synthesis";     Config = "configs/rmr_research/gen7/g7_champion_synthesis.yaml" },

    # 4. Hyperparameter Sweeps & Sensitivity Controls
    @{ RunId = "g7_h1_fidt_k2";             Config = "configs/rmr_research/gen7/g7_h1_fidt_k2.yaml" },
    @{ RunId = "g7_h1_fidt_k6";             Config = "configs/rmr_research/gen7/g7_h1_fidt_k6.yaml" },
    @{ RunId = "g7_h2_chfl_lam02";          Config = "configs/rmr_research/gen7/g7_h2_chfl_lam02.yaml" },
    @{ RunId = "g7_h2_chfl_lam10";          Config = "configs/rmr_research/gen7/g7_h2_chfl_lam10.yaml" }
)

$logDir = "runs/sha_a/suite_logs_gen7"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RMR GEN 7 COMPREHENSIVE FACTORIAL MATRIX & BREAKTHROUGH SUITE (PowerShell)" -ForegroundColor Cyan
Write-Host "  17 Experiments | Target: Breakthrough Sub-60 MAE (< 60.0)" -ForegroundColor Cyan
Write-Host "  Constraints: <= 105,000 params, Zero KD, ShanghaiTech Part A" -ForegroundColor Cyan
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
        $evalLog = "$logDir/${runId}_eval_direct.log"
        & $py -m rmr_v3.eval --checkpoint $ckptPath --no-tiling 2>&1 | Tee-Object -FilePath $evalLog

        Write-Host ">>> Evaluating Horizontal Flip TTA: $ckptPath" -ForegroundColor Magenta
        $evalTtaLog = "$logDir/${runId}_eval_tta.log"
        & $py -m rmr_v3.eval --checkpoint $ckptPath --tta --no-tiling 2>&1 | Tee-Object -FilePath $evalTtaLog
    }
}

Write-Host "`n>>> Summarizing Gen 7 Suite Results:" -ForegroundColor Cyan
& $py scripts/summarize_gen7_suite.py

Write-Host "`nAll Gen 7 experiments completed!" -ForegroundColor Cyan
