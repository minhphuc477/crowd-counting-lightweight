# GEMINI.md — RMRv3 Research & Engineering Reference

## 1. Execution Environment
- **Target Workstation GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM) with PyTorch 2.6+ CUDA 12.4 and AMP FP16.
- **Local Workspace**: `f:\lightweightcrcn` (Windows PowerShell `.venv\Scripts\python.exe`, or `python` on Linux GPU workstation).
- **Git Branch**: `RMR` (`origin/RMR`), governed by root [`AGENTS.md`](file:///f:/lightweightcrcn/AGENTS.md) and [`rules/RULES.md`](file:///f:/lightweightcrcn/rules/RULES.md).
- **Entrypoints**:
  - Training: `python -m rmr_v3.trainer --config <yaml_path>`
  - Evaluation: `python -m rmr_v3.eval --config <yaml_path> --checkpoint <ckpt_path>`

## 2. Research Scope & Survey-Anchored Methodology
- **Topic**: Constrained Radon Measure Recovery (`rmr_v3`) for sub-$105k$ standalone ($0\%$ Knowledge Distillation) crowd counting across ShanghaiTech Part A/B, UCF-QNRF, JHU-Crowd++, and UCF_CC_50.
- **Survey Grounding**: All research gaps, hypotheses, and baseline comparisons follow the 4-step framework in [`docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md`](file:///f:/lightweightcrcn/docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md), anchored in 2024–2025 surveys (*Wang et al., IET Image Processing 2025*; *Cheng et al., Meas. Sci. Technol. 2025*; *Gao et al., Vicinagearth 2025*).

## 3. Non-Negotiable Engineering & Scientific Invariants
1. **Strict Parameter Ceiling & 0% KD**: All primary models must have $\le 104,441$ trainable parameters and $0.0\%$ Knowledge Distillation.
2. **Dataset-Agnostic Design**: Zero dataset-specific architecture hacks, hardcoded logit thresholds, or per-dataset tuning in the unified `configs/benchmarks/nextgen_sub65/` suite (`crop_size: 512`, `init_m0: auto`, shared losses and optimizer schedule). Both [`rmr_v3/trainer.py`](file:///f:/lightweightcrcn/rmr_v3/trainer.py) and [`rmr_v3/engine.py`](file:///f:/lightweightcrcn/rmr_v3/engine.py) pass unmodified `compute_manifest_density(..., output_stride=4)` into `FineMeasureHead`, which handles `subpixel_stride2` scaling internally.
3. **Banned Terminology (`AGENTS.md` Rule 1)**: Never use the banned multi-objective evolutionary buzzword prohibited by `AGENTS.md` Rule 1 in any file, plot, table, or response. Always use *"Empirical Complexity-Accuracy Trade-off"*.
4. **No Automation Scripts (`AGENTS.md` Rule 2)**: Never create or modify `run_all.sh`, `train_all.ps1`, `batch_runner.py`, etc. Provide standalone copy-paste CLI commands only.
5. **Empirical Integrity & Anti-Hype (`AGENTS.md` Rule 3)**: Report metrics strictly from actual files (`summary.json`, `attribute_breakdown.json`, `results/*.json`, verified papers). Explicitly disclose Phantom Background Fog (`65.57%` off-peak mass on SHA), Softplus bias freeze ($b \approx -5.442$), extreme-density undercounting (`132.16` Dense MAE on SHA Pretrained), distractor false positives (`185.10` phantom count on 4 JHU distractors), and zero-shot cross-dataset drops.
6. **Verified Peer-Reviewed Baselines**: Anchor sub-$200k$ standalone comparisons against verified papers (`TinyCount` 60k -> 78.2 SHA, `Lw-Count` 71k -> 69.7 SHA, `MCNN` 133k -> 110.2 SHA, `LSANet` 200k -> 66.1 SHA / 8.6 SHB) and larger references (`EBC-ZIP-P` 0.81M -> 71.18 SHA, `EBC-ZIP-N` 3.36M -> 58.86, `EBC-ZIP-S` 33.60M -> 55.17, `EBC-ZIP-B` 105.60M -> 47.81). Never cite fabricated `84k` params for `EBC-ZIP-S`.
7. **Continuous Git Sync**: Commit and push all verified changes to `origin/RMR`.

## 4. Authoritative Repository Documentation
- [`README.md`](file:///f:/lightweightcrcn/README.md) — Repository overview and quickstart.
- [`docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md`](file:///f:/lightweightcrcn/docs/SCIENTIFIC_FRAMEWORK_AND_PAPER_ALIGNMENT_MATRIX.md) — 4-step survey framework, RQs/ROs, baselines, empirical tables, and failure modes.
- [`docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md`](file:///f:/lightweightcrcn/docs/CODEBASE_ENTRYPOINTS_AND_OPERATIONS_MANUAL.md) — Architecture entrypoints and standalone training/eval CLI commands for `configs/benchmarks/nextgen_sub65/`.
- [`model_facts.md`](file:///f:/lightweightcrcn/model_facts.md) — Exact module parameter counts and component audits.
- [`input_contract.md`](file:///f:/lightweightcrcn/input_contract.md) — Falsifiable hypotheses and negative result archive.
- [`results/forensics/full_182_forensic_results.json`](file:///f:/lightweightcrcn/results/forensics/full_182_forensic_results.json) — 182-image mass decomposition forensics.
