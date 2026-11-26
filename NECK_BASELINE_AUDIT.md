# NECK_BASELINE_AUDIT.md — Official Code Audit vs. `rmr_core/necks/`

Code-faithful audit comparing **6 official open-source FPN neck implementations** (verified directly from GitHub) against every neck in [`rmr_core/necks/`](file:///f:/lightweightcrcn/rmr_core/necks/__init__.py) and our **SHA Gate-1 empirical results** (`Pure Bayesian 1.0 + Anscombe 1.0 + L1 0.02`, shared `MobileNetV4-S050` backbone + `DSIMR` head).

---

## 1. Official Published Baseline Necks (Verified From Official Repos)

| Official Neck | Repo & Source File | Lateral / Upsample | Bottom-Up / Downsample | Fusion Op | Post-Fusion & Norm | Top-Level Context | BU Feedback to Lowest-Stride (`P_min_out`) | Fast-Normalized Convex Fusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **FPN** (*CVPR 2017*) | [`open-mmlab/mmdetection/fpn.py`](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/models/necks/fpn.py) | `1x1 Conv` / `nearest` | None (`MaxPool2d(1,s=2)` for extra levels) | Plain `+` | `3x3 Conv` (`norm_cfg=None` default) | **No** | **No** | **No** |
| **PAFPN** (*CVPR 2018*) | [`open-mmlab/mmdetection/pafpn.py`](https://raw.githubusercontent.com/open-mmlab/mmdetection/main/mmdet/models/necks/pafpn.py) | `1x1 Conv` / `nearest` | `3x3 s=2 p=1 Conv` (`inter_outs[i]` $\to$ `i+1`) | Plain `+` (TD & BU) | `outs[0] = inter_outs[0]`; `outs[1:] = 3x3 Conv` | **No** | **No** (`outs[0]` skips BU path) | **No** |
| **BiFPN** (*CVPR 2020*) | [`google/automl/efficientdet_arch.py`](https://raw.githubusercontent.com/google/automl/master/efficientdet/efficientdet_arch.py) | `1x1 Conv` / `nearest` | `MaxPool2d(k=3, s=2)` + `P_in` skip edge | `sum(w_i * x_i) / (sum(w) + 1e-4)` (`ReLU(w)`) | `Swish -> SepConv3x3 -> BN` | **No** | **No** (single cell: `P_min_out` ends TD pass) | **Yes** (`'fastattn'` D0–D5) |
| **AFPN** (*2023*) | [`gyyang23/AFPN/afpn.py`](https://raw.githubusercontent.com/gyyang23/AFPN/master/mmdetection/mmdet/models/necks/afpn.py) | `1x1 BasicConv` (`C//8`) / `bilinear` | `BasicConv(k=s, s=s, p=0)` (`s in {2,4,8}`) | `ASFF_{2,3,4}` (`Softmax(dim=1)` spatial weights) | `4 x BasicBlock` (`3x3 Conv+BN+ReLU`) | **No** | **Yes** (`scalezero` receives `x1,x2,x3` via `ASFF`) | **No** (`Softmax` spatial) |
| **FreqFusion** (*TPAMI 2024*) | [`Linwei-Chen/FreqFusion.py`](https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/main/FreqFusion.py) | `1x1` / **ALPF** (`3x3` $\to$ `Softmax*Hamming 5x5` $\to$ `CARAFE`) | None (TD only; optional `LocalSimGuidedSampler` `grid_sample`) | **AHPF** (`hr - carafe(hr, mask_hr) + hr`) + plain `+` | `3x3 Conv`; `GN` in sampler | **No** | **No** | **No** (`Softmax*Hamming`, plain `+`) |
| **HS-FPN** (*AAAI 2025*) | [`ShiZican/HS-FPN/hs_fpn.py`](https://raw.githubusercontent.com/ShiZican/HS-FPN/main/hs_fpn.py) | `1x1 Conv` / `nearest` | None (TD only) | **SDP** cross-attn + plain `+` | **HFP** (`DCT` HP + `GAP/GMP` + `3x3 Conv + GN(32)`) | **Partial** (`HFP` self-attn on `P4`) | **No** | **No** |

### The Single-Head Crowd Counting Dead-Gradient Trap
In multi-scale object detectors (MMDetection / EfficientDet), detection heads attach to **every** pyramid level (`P3..P7`), so bottom-up layers in `PAFPN` and `BiFPN` receive gradients from coarse heads (`P4'', P5'', ...`). In **single-head Stride-4 crowd counting** (`y0 = FineHead(P4_out)`), `P8_out` and `P16_out` have **no attached head**. Under official `PAFPN` (`outs[0] = inter_outs[0]`) or a single-cell `BiFPN` (`P_min_out` finalized in top-down), **100% of the bottom-up pathway receives zero gradient** unless adapted with a feedback consensus node into `P4_out` (like `AFPN`'s `ASFF` or our `fuse_p4`).

### Paper vs. Official GitHub Code Nuances (Verified via arXiv)
1. **PANet (`arXiv:1803.01534`):** Paper confirms $N_2 = P_2$ inside the pyramid, while adding **Adaptive Feature Pooling** (RoIAlign across all $\{N_2..N_5\}$ + element-wise max/sum) inside the RoI head (omitted from MMDetection's standalone `pafpn.py`).
2. **BiFPN (`arXiv:1911.09070`):** Paper stacks $D_{\text{bifpn}} \in [3, 8]$ repeated cells; when stacked $\ge 2$ times, $P_3^{(2)}$ receives bottom-up context from layer 1 ($P_4^{(1)} \to P_3^{(2)}$), whereas a single cell ($L=1$) ends $P_{\min}$ in top-down.
3. **FreqFusion (`arXiv:2408.12879`):** Paper presents the **Offset Generator** (`feature_resample=True`) as one of the 3 core active modules (ALPF + Offset Generator + AHPF), whereas `FreqFusion.py` sets `feature_resample=False` as constructor default; `hamming_2d` windowing is applied before kernel normalization in code.
4. **FPN (`arXiv:1612.03144`), AFPN (`arXiv:2306.15988`), HS-FPN (`arXiv:2412.10116`):** Paper text matches official GitHub implementations.

---

## 2. Exact Audit of `rmr_core/necks/` vs. Original Papers

| Local Class (`rmr_core/necks/`) | Total Params (w/ DSIMR) | Faithful to Original? | Exact Deviations from Official Paper |
| :--- | :---: | :---: | :--- |
| **`AdditiveFPNNeck`** ([`additive.py`](file:///f:/lightweightcrcn/rmr_core/necks/additive.py)) | `98,870` | **Modified** | Adds 3 parallel `DepthwiseDilated(1,2,3)` on `L16` + `DSResidual(GN+SiLU)` + bilinear upsampling (vanilla 2017 FPN has no `L16` dilated context and uses plain `3x3` + nearest). |
| **`BiFPNLiteNeck`** ([`bifpn.py`](file:///f:/lightweightcrcn/rmr_core/necks/bifpn.py)) | `99,268` | **Faithful + Adapted** | Keeps BiFPN fast-normalized `ReLU(w)/(sum+1e-4)` fusion, `MaxPool2d(3,s=2)`, and zero top-level context (`use_gap=False`). Adapts `GN+DSResidual` and adds 4-way `w_out4` into `p4_out` so bottom-up layers receive gradients in single-head counting. |
| **`PAFPNBiNormNeck`** ([`pafpn.py`](file:///f:/lightweightcrcn/rmr_core/necks/pafpn.py)) | `100,647` | **Custom Architecture (NOT vanilla PAFPN!)** | **Differs fundamentally from Liu et al. (CVPR 2018):** (1) adds `StripMultiKernelContext (3x3, 1x5, 5x1)` on `L16`, (2) replaces plain `+` with `ConvexFastNormalizedFusion` at all 6 nodes, (3) adds `Residual(DW3x3+GN)` at `p4_td`, (4) uses learnable `DWConv3x3(s=2)+GN+SiLU` downsampling, (5) adds 4-way `fuse_p4([l4, p4_td, up(p8_out), up(p16_out)])`. **Never cite/label this as "vanilla PAFPN baseline".** |
| **`ASPPLiteFPNNeck`** ([`aspp.py`](file:///f:/lightweightcrcn/rmr_core/necks/aspp.py)) | — | **Custom Lite** | Parallel `DepthwiseDilated(1,3,6)` + optional `GAP` + `1x1` proj + top-down `DSResidual`. |
| **`HDCLiteFPNNeck`** ([`hdc.py`](file:///f:/lightweightcrcn/rmr_core/necks/hdc.py)) | — | **Custom Lite** | Sequential `DepthwiseDilated(1,2,3)` cascade on `L16` + `1x1` proj + top-down `DSResidual`. |
| **`SFILiteFPNNeck` (`LFI-Lite`)** ([`sfi.py`](file:///f:/lightweightcrcn/rmr_core/necks/sfi.py)) | `99,766` | **Proposed** | `HDCLite` (`groups=2` proj) + norm-free linear Laplacian HF residual `gamma_hf * DW3x3(L4 - AvgPool3x3(L4))` (`+320` params, WD-exempt). |
| **`LFIS2FPNNeck` (`LFI-S2`)** ([`lfi_s2.py`](file:///f:/lightweightcrcn/rmr_core/necks/lfi_s2.py)) | `99,314` | **Proposed (Parked)** | Adds `C2` (`H/2`) skip + foldable Stride-2 `BatchNorm2d` + `NativeS2DSIMRHead`. |

---

## 3. SHA Gate-1 Empirical Results (`MobileNetV4-S050 + DSIMR + Pure Bayesian + Anscombe + L1`, `1500` epochs)

| Neck Variant | Params | Seed | Best MAE / RMSE | Final EMA (`ep=1500`) | GAME2 | Architectural Takeaway |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`LFI-Lite` (`laplacian`)** | `99,766` | `42` | `63.08 / 100.78` | `63.52 / 100.79` | `88.86` | **3-seed mean: `64.78 ± 2.32` (Best) / `66.18 ± 4.15` (EMA)** — edges out `PAFPNBiNormNeck` on 3-seed average with 881 fewer params. |
| **`LFI-Lite` (`laplacian`)** | `99,766` | `123` | `63.84 / 101.77` | `64.06 / 102.69` | `89.08` | — |
| **`LFI-Lite` (`laplacian`)** | `99,766` | `2025` | `67.42 / 105.77` | `70.97 / 109.20` | `91.07` | — |
| **`PAFPNBiNormNeck`** *(Strip-Context Bi-Convex)* | `100,647` | `42` | **`62.68 / 98.02`** | **`62.69 / 98.01`** | **`86.37`** | **3-seed mean: `65.21 ± 2.36` (Best) / `66.86 ± 3.70` (EMA)** — same `±2.3` SHA seed variance; seed 42 was a favorable outlier. |
| **`PAFPNBiNormNeck`** *(Strip-Context Bi-Convex)* | `100,647` | `123` | `65.60 / 103.04` | `68.18 / 107.53` | `88.52` | — |
| **`PAFPNBiNormNeck`** *(Strip-Context Bi-Convex)* | `100,647` | `2025` | `67.35 / 104.35` | `69.72 / 110.52` | `92.36` | — |
| **`LFI (direct_dw)`** *(No AvgPool subtraction)* | `99,766` | `42` | `64.05 / 98.71` | `65.48 / 103.31` | `88.37` | Within 3-seed `std` of `LFI-Lite` (`98.71` RMSE < `100.78`), falsifying strict Laplacian-only necessity. |
| **`AdditiveFPNNeck`** | `98,870` | `42` | `66.16 / 101.91` | `67.02 / 102.54` | `89.38` | Dilated `L16` + single-path TD lacks Stride-4 capacity & BU feedback. |
| **`BiFPNLiteNeck`** | `99,268` | `42` | `67.07 / 102.08` | `68.12 / 105.71` | `90.21` | Same bidirectional graph as `PAFPNBiNormNeck`, but lacks `StripMultiKernelContext`, `td4_dw`, and learnable `s=2` DW downsample. |
| **`LFI (no_lap)`** | `99,446` | `42` | `68.44 / 106.29` | `70.95 / 108.43` | `93.11` | Single-pass `L4 + up(P8)` with no `td4` refinement suffers severe `P4` degradation. |

### Paper Naming & Positioning Guidance
1. **Do NOT label `PAFPNBiNormNeck` as "PAFPN (CVPR 2018)"** — doing so misattributes a custom Strip-Context Bidirectional Convex FPN (`100.6k`) as a generic baseline.
2. **3-Seed Conclusion:** Neither `LFI-Lite` (`64.78 ± 2.32`) nor `PAFPNBiNormNeck` (`65.21 ± 2.36`) dominates the other beyond SHA's `~2.3` MAE 3-seed variance, and `LFI-Lite` actually achieves a slightly lower 3-seed mean (`-0.43` MAE best / `-0.68` MAE EMA) with fewer parameters (`99.8k` vs `100.6k`) and simpler top-down structure. However, because `direct_dw` (`64.05` on seed 42) also falls inside this `±2.3` band, any SHA claim must be validated on larger datasets (`SHB`, `QNRF`) or via multi-seed `direct_dw` before claiming Laplacian-specific superiority.
