# RESEARCH_CONTRACT.md — Target Paper & Architecture Spec

## 1. Publication Target & Hard Budgets
- **Venue:** *Journal of Real-Time Image Processing* (JRTIP, Springer — SJR Q1 / JCR Q2).
- **Problem:** Constrained Radon Measure Recovery for real-time sub-100k crowd counting (`<= 100k` trainable params canonical target, `<= 104,441` hard ceiling, `0%` Knowledge Distillation, pure ONNX-exportable ops).
- **Primary Architecture Baseline:** `LFI-Lite + DSIMR` (`99,766` params, single-pass Direct Fast-Path).
- **Active Cure Candidates:** `D0.5` (Unbounded HF + Pad-Aware), `D1` (+ Anscombe/L1 count loss), `D2` (`LFI-S2` `99,314` params preserving Backbone `C2` Stride-2 + foldable Stride-2 `BatchNorm2d`).

## 2. Core Research Gaps to Solve for the Paper
1. **Ultra-Dense Undercounting (`GT >= 3000`):** Current `LFI-99k` suffers `-32.9%` RelBias on `NWPU 3000+` (`n=8`, MAE `2190.50`) and `-30.3%` on `JHU Test 3000+` (`n=25`, MAE `1507.47`) caused by:
   - Stride-4 backbone downsampling dropping `C2` (`32ch @ H/2`),
   - `tanh` HF amplitude clipping in `LFI-Lite`,
   - NB count loss `1/y` gradient decay at high counts,
   - Input resolution cap (`max_size=2048`) crushing small heads before inference.
2. **Real-Time Edge Efficiency vs. Accuracy:** Must achieve real-time/high-FPS CPU ONNX & GPU FP16 throughput (`>= 25–40+ FPS` @ 512 CPU 1-thread ORT, high CUDA FPS at `720p/1080p`) without sacrificing dense accuracy.
3. **Empty-Scene / Sparse Fog:** Reduce background mass leakage (`NWPU ZeroMAE 9.72`, `JHU Test Zero n=4 MAE 70.45`) without introducing dead gates.

## 3. Keep vs. Discard Decision Criterion (Autoresearch Style)
- **KEEP** a change only if it strictly improves MAE/RMSE or `dense_3000_rel_bias_pct` at matched/lower params (`<100k`) without breaking real-time ONNX deployment or regressing `ZeroMAE`.
- **DISCARD** any change that increases params `>= 104k`, breaks ONNX export, introduces confounded multi-variable gains without ablation proof, or degrades CPU/GPU FPS unjustifiably for negligible MAE gain.

## 4. Required Evidence Checklist Before Paper Writing
- [x] `LFI-99k` 5-benchmark baseline (`SHA`, `SHB`, `QNRF`, `NWPU Val`, `JHU Test`) + 3 zero-shot evals.
- [ ] Dense-cure causal ladder on `SHA` & `NWPU` (`D0.5` -> `D1` -> `D2`) + `D3` tiling vs `2048` downscale eval.
- [ ] Freeze winning `<100k` architecture -> complete all 5 benchmarks + 3 zero-shot if changed from `LFI-99k`.
- [ ] Clean single-variable ablations on the winning model (`No-Laplacian`, `HF tanh vs linear`, `NB vs Anscombe+L1`, `S2 C2 skip`).
- [ ] GPU CUDA FP32/FP16 + CPU 1-thread ONNX Runtime latency (`512`, `720p`, `1080p`), VRAM, FP16 counting parity, and 4-panel qualitative figures.
