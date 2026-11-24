# AGENTS.md — Repository Control Rules (Lean SSOT)

## 1. Source of Truth & Scope Boundaries
- **Read FIRST:** `RESEARCH_CONTRACT.md`, `DEAD_ENDS.md`, `EXPERIMENTS.tsv`. Ignore legacy `.md` reports/plans in the repo.
- **Primary Focus:** Model architecture improvements (`rmr_core/necks/`, `rmr_core/heads.py`, `rmr_core/native_s2_head.py`, `rmr_core/backbones.py`, `rmr_v3/losses/`, `configs/benchmarks/`) and paper deliverables.
- **Strictly Off-Limits (Unless User Asks):** Unsolicited code cleanups, style refactors, tech-debt sweeps, redundant wrappers, or new abstractions. Do not modify metric/evaluation code to inflate numbers.
- **450-Line Hard Limit:** Keep every `.py` file under `rmr_core/` and `rmr_v3/` strictly `<= 450` lines.

## 2. Anti-Drift & Token-Economy Rules
- **Concise Answers Only:** State root cause -> exact architectural/math change -> verification command/result. No filler, no repeating prior context.
- **Quiet Verification:** Run targeted `pytest -q` and grep/print summary metrics only; never flood context with verbose logs.
- **Simplicity Gate:** Complexity is guilty until proven innocent. A change that adds complexity without clear metric/runtime gain must be discarded.
- **Fail-Closed & 2-Correction Stop:** If a canonical command fails or the user corrects the same drift twice, stop and report the exact blocker; never invent workarounds.

## 3. Non-Negotiable Repository Constraints
1. **ABSOLUTE BAN ON THE P-WORD ("PARETO"):** Strictly prohibited across all titles, prose, plots, tables, code comments, and agent replies. Use *"Empirical Complexity-Accuracy Trade-off Plot"* / *"Biểu đồ đánh đổi thực nghiệm độ phức tạp - độ chính xác"*.
2. **STRICT SCRIPT IMMUTABILITY:** NEVER create or execute batch runner scripts (`run_all.sh`, `train_all.ps1`, `batch_runner.py`). Output standalone copy-pasteable commands only.
3. **ZERO HYPE & FAILURE-MODE HONESTY:** Never use marketing buzzwords. Report exact measured logs/numbers only and explicitly disclose failure modes (background fog, Softplus bias freeze, `3000+` dense undercounting, latency regressions).
4. **CONTINUOUS GIT SYNC:** Every edit to code, configs, or docs must be committed and pushed immediately to `origin/RMR`.
