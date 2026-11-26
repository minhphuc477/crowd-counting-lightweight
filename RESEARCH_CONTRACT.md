# RESEARCH_CONTRACT.md — JVCI Target Spec & Gate-1 Validation

## 1. Publication Target & Active Canonical Architecture
- **Primary Venue:** *Journal of Visual Communication and Image Representation* (JVCI, Elsevier — SCIE Q2, Subscription free; backup: *Pattern Recognition Letters* / *Neurocomputing*).
- **Hardware Norm:** Standard single-GPU/CPU profiling (`Params`, `GFLOPs`, `ms/FPS`, `VRAM` with explicit input size, threads, precision, warm-up/repeats).
- **Active Architecture:** **`LFI-99k`** (`99,766` params, `0%` KD, `MobileNetV4-S050 + LFI-Lite + DSIMR`, single-pass Direct Fast-Path, ONNX opset-17).
- **Active Loss (Updated):** **Pure Bayesian (`1.0`) + Anscombe (`1.0`) + Count L1 (`0.02`)** (`Radon-OT` dropped after verified `+7.12` MAE regression on SHA: `70.20` vs `63.08` best; `70.92` vs `63.52` final EMA).
- **Checkpoint Protocol (Zero Test-Set Peeking):** Report **Final EMA (`last.pt`, epoch 1500)** as primary leak-free metric (`63.52` vs `68.53` verified on SHA), alongside `best_val_mae` transparency where applicable.

## 2. Gate-1 Go/No-Go on SHA (Same Head `DSIMR`, Same Pure Bayesian Loss)
Every claim below is falsifiable. Do NOT expand to SHB/QNRF/NWPU/multi-backbone until **Gate-1** passes on SHA (`mean ± std` across 3 seeds `42, 123, 2025`):
- **H1 (Neck Gain vs Fair Baselines):** `LFI-Lite + DSIMR` (`99,766` p) beats `Additive FPN + DSIMR` (`98,870` p), `BiFPN-Lite + DSIMR` (`99,268` p), and `PAFPN + DSIMR` (`100,647` p) by `> 3-seed std` on SHA.
- **H2a (Laplacian HF Control — Critical):** `LFI-Lite` (`laplacian`) beats `no_laplacian` (`99,446` p), `fixed_hf` (`99,446` p), and **`direct_dw` (`99,766` p, same DWConv3x3 without AvgPool subtraction)** by `> 3-seed std` on SHA. If `direct_dw` matches `laplacian`, pivot framing immediately.
- **H2b (Negative-Result Evidence):** Retain verified failure-mode controls (`selective_gate`, `tanh`, `Radon-OT`, and `LFI-S2` Stride-2 `BN` drift `ep=225 -> 1295`) as quantitative empirical evidence.
- **H3 (Multi-Seed Rigor):** Report `mean ± std` across 3 seeds (`42`, `123`, `2025`) on SHA before finalizing claims.

## 3. Required Evidence Checklist
- [x] Data leakage audit (`0` exact path overlap across all 5 datasets) + Snowballing literature audit (`LITERATURE_REVIEW.md`).
- [ ] **Gate-1 on SHA:** `Additive+DSIMR`, `BiFPN+DSIMR`, `PAFPN+DSIMR`, `no_laplacian`, `direct_dw`, `fixed_hf`, and `seed123`/`seed2025` for `LFI-Lite+DSIMR` (all on Pure Bayesian + Anscombe + L1).
- [ ] **Gate-2 Expansion (only if Gate-1 passes):** Plug-and-play second backbone (`ResNet18` / `MobileNetV3`), `SHB`, `QNRF`, `NWPU`/`JHU`, cross-dataset (`SHA->SHB`, `QNRF->SHA/SHB`), density-bin breakdown, and `L4 / L4_hf / P4` visualizations.
