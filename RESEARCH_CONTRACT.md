# RESEARCH_CONTRACT.md — JVCI Target Spec & Conditional Hypotheses

## 1. Publication Target & Frozen Paper Architecture
- **Primary Venue:** *Journal of Visual Communication and Image Representation* (JVCI, Elsevier — SCIE Q2, Subscription free; backup: *The Visual Computer*, Springer).
- **Hardware Norm:** Standard single-GPU/CPU profiling (`Params`, `GFLOPs`, `ms/FPS`, `VRAM`); no Jetson/Raspberry Pi required.
- **Frozen Active Architecture:** **`LFI-99k`** (`99,766` params, `0%` KD, `MobileNetV4-S050 + LFI-Lite + DSIMR`, single-pass Direct Fast-Path, ONNX opset-17).
- **Parked Branches:** `LFI-S2 / D0.5 / D1 / D2` remain deferred unless `H1/H2` fails or future extension is needed.

## 2. Conditional Hypotheses (Keep Claim ONLY If Data Supports It)
Every claim below is a falsifiable hypothesis. If an ablation shows a component adds no gain above multi-seed variance, **drop that claim**.
- **H1 (Fair Same-Budget Gain):** `LFI-99k` outperforms `fair_baseline_sha_additive_99k` (`Additive FPN + DW-SubPixel + Pure Bayesian`, same backbone/pipeline/epochs) by more than 3-seed `std` on `SHA`.
- **H2a (Laplacian HF Claim):** Keep only if `LFI-99k` beats `abl_sha_lfi_no_laplacian_99k` by `> 3-seed std` on `SHA`.
- **H2b (DSIMR Support Gate Claim):** Keep only if `LFI-99k` improves `ZeroMAE` / `RMSE` over `abl_nwpu_lfi_dw_subpixel_99k` on `NWPU Val`.
- **H2c (Radon-OT Claim):** Keep only if `LFI-99k` beats `abl_sha_lfi_pure_bayesian_99k` by `> 3-seed std` on `SHA`.
- **H3 (Multi-Seed Rigor):** Report `mean ± std` across 3 seeds (`42`, `123`, `2025`) on `SHA` before finalizing claims.
- **H4 (Transparent Limitations):** Explicitly disclose Stride-4 Nyquist + `max_size=2048` undercounting on `NWPU/JHU 3000+` (quantified via `D3` tiling eval) and `SHA/SHB/QNRF` test-split checkpoint selection vs blind `JHU Test`.

## 3. Required JVCI Evidence Checklist
- [x] Data leakage audit (`0` exact path overlap across all 5 datasets) + `LFI-99k` 5 benchmarks + 3 zero-shot evals (`seed=42`).
- [ ] Run `1` fair baseline + `3` ablations + `2` extra SHA seeds (`6` runs total).
- [ ] Run `0`-train evals: `D3` (`eval_nwpu_tiling_sweep.py`), `eval_sfi_mechanisms.py` (SHA), GPU FP32/FP16 profile, and 4-panel qualitative + feature-map figures.
- [ ] Manually verify every external paper citation, param count, and Val/Test split in `LITERATURE_REVIEW.md` before drafting.
