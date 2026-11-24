# LITERATURE_REVIEW.md — Prior Art, Diagnostic Proof & Architecture Connections

Citation-verified literature map linking published crowd counting findings (2018–2026) directly to our `<100k`-parameter architecture, failure modes, and diagnostic experiments.

---

## 1. Sub-200k & Real-Time Lightweight Models (Scope & Aim Comparison)

| Model | Venue / Year | Params | Output Stride | Core Architecture / Loss | SHA / QNRF / NWPU Val | Connection to Our Architecture |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **LCDnet** (Khan et al.) | *JRTIP* 2023 ([arXiv:2302.05374](https://arxiv.org/abs/2302.05374)) | `50k` | **S2 (`1/2`)** | 6-layer CNN, asymmetric `1x3`+`3x1` convs; MSE + curriculum | `118.4` / — / — | Proves `1/2` stride preserves tiny targets, but plain 6-layer CNN lacks semantic context (`SHA 118.4`). |
| **TinyCount** (Lee & Lee) | *JRTIP* 2024 ([Springer](https://link.springer.com/article/10.1007/s11554-024-01531-8)) | `60,078` | S8 $\to$ Deconv | MobileNetV2 + SE + dilated SPM + TransposedConv; MSE | `80.3` / `143.1` / — | Primary JRTIP baseline; S8 bottleneck loses dense QNRF heads (`143.1`). |
| **Lw-Count** (Liu et al.) | *IEEE TCSVT* 2022 ([DOI](https://doi.org/10.1109/tcsvt.2022.3171235)) | `70k` | Enc-Dec | Ghost ELCM + Spatial GroupNorm (SGN) + SRM; RNCC + Count | `69.7` / `149.7` / — | Motivates `GroupNorm`/SGN for uneven empty-to-dense variance. |
| **C-CNN** (Shi et al.) | *ICASSP* 2020 ([arXiv:2002.06515](https://arxiv.org/abs/2002.06515)) | `73k` | S8 | 3 parallel front filters (`5,7,9`) + 6-layer trunk; MSE | `88.1` / — / — | Shows multi-scale front kernels help (`88.1` vs `98.9`), still limited by S8 pooling. |
| **EdgeCount** (Shen et al.) | *IEEE TCSVT* 2024 ([DOI](https://doi.org/10.1109/TCSVT.2024.3469933)) | `120k` | Enc-Dec | MobileViT + SCConv (SRU+CRU) + LWMFFM; **DMKD distillation** | — | Not standalone (`DMKD`); whereas ours uses `0% KD`. |
| **CPMS-Net** (Shang et al.) | *JKSU-CIS* 2026 ([DOI](https://doi.org/10.1007/s44443-026-01052-3)) | `125,132` | S8 $\to$ 3-branch | MobileNetV2 + CPM + MSOM + LRAM ($H/32$ MHA); **RCJL (`MSE + 4x4 grid L1`)** | `72.3` / `126.7` / — | Validates regional `4x4` grid `L1` count conservation against dense undercounting. |
| **DroneNet** (Khan et al.) | *IEEE CCNC* 2023 ([arXiv:2211.07137](https://arxiv.org/abs/2211.07137)) | `150k` | S4 | 3-col Self-ONN Maclaurin nodal ops + Tanh; MSE | `98.5` / — / — | Tanh-bounded nonlinearities limit extreme high-density scaling. |
| **Zhao et al.** | [arXiv:2510.13250](https://arxiv.org/html/2510.13250v1) (2025) | `149k` | S4 | From scratch; `(9,7,5)` stem + ShuffleNetV2 + Lite-HRNet CCW/MLF + `3x3` FPN; MSE (`512x1024`) | `63.4` / `104.1` / `102.0` | Strongest sub-150k prior baseline; large continuous kernels beat dilated `3x3` (`104.1` vs `110.1`), yet S4 + `512x1024` still undercounts ultra-dense scenes. |
| **MobileCount** (Wang et al.) | *Neurocomputing* 2020 ([GitHub](https://github.com/ChenyuGAO-CS/MobileCount)) | `3.40M` | S32 $\to$ S4 | MobileNetV2 + Light-Weight RefineNet (CRP); MSE | `89.4` / `131.1` / — | Proves heavy 3.4M S32 encoder still lags compact high-res designs. |
| **LFI-99k (Ours Base)** | This work (`5418c31`) | **`99,766`** | **S2 (subpixel)** | MobileNetV4-S050 (`C4,8,16`) + HDC `(1,2,3)` + Laplacian HF + DSIMR | **`64.05` / `94.71` / `81.21`** | Beats TinyCount/CPMS-Net/Lw-Count/Zhao-149k on QNRF & NWPU at `<100k`, `0% KD`. |
| **LFI-S2 (Ours D2)** | This work (`05dc71e4`) | **`99,314`** | **S2 (native `C2`)** | Preserves `C2 (32ch @ H/2)` + Unbounded Laplacian HF + Foldable S2 BN + Native S2 DSIMR | *Pending D2* | Directly targets S4 Nyquist aliasing (`3000+` undercounting). |

---

## 2. Gap 1 Proof — Resolution Collapse & Nyquist/Scale Truncation in Dense Crowds

- **LSC-CNN** (Sam et al., *IEEE TPAMI*, [ar5iv:1906.07538](https://ar5iv.labs.arxiv.org/html/1906.07538)): Diagnoses that `1/8` and `1/16` feature strides merge dense tiny heads into unresolvable blobs; introduces top-down branches down to **`1/2` input resolution** to separate packed heads.
- **P2PNet** (Song et al., *ICCV 2021*, [ar5iv:2107.12746](https://ar5iv.labs.arxiv.org/html/2107.12746)): Table 5 (SHA stride ablation) proves localization `nAP_delta` collapses across coarse strides: **`66.8 (s=4)` $\to$ `64.4 (s=8)` $\to$ `52.4 (s=16)`**, while MAE degrades (`53.51 / 52.74 -> 54.3`).
- **SASNet** (Song et al., *AAAI 2021*, [DOI](https://doi.org/10.1609/aaai.v35i3.16360)) & **FIDT** (Liang et al., *IEEE TMM 2022*, [ar5iv:2102.07925](https://ar5iv.labs.arxiv.org/html/2102.07925)): Both prove Gaussian blobs + coarse strides merge dense local maxima; SASNet routes tiny heads to `1/2` & `1/1` decoder stages, while FIDT keeps HRNet high-resolution maps.
- **SACC-Net** (Hsieh et al., *arXiv:2312.16771 / IEEE TIP 2025*, [HTML](https://arxiv.org/html/2312.16771)): Explicitly names **scale truncation** across standard `1/2 -> 1/4 -> 1/8` stride pyramids as the cause of tiny-head disappearance.
- **STEERER** (Han et al., *ICCV 2023*, [arXiv:2308.10468](https://arxiv.org/html/2308.10468)): Uses selective inheritance from low-to-high resolution (HRNet FSIA) to avoid cross-scale smearing (`NWPU test MAE 63.7`).
- **Connection to Ours:** Confirms `LFI-99k`'s drop from QNRF (`-3.1%` RelBias at `3000+`) to NWPU (`-32.9%` at `3000+`) and JHU (`-30.3%` at `3000+`) is fundamentally **Stride-4 Nyquist aliasing** (`C2` discarded by `TimmPyramidBackbone`), directly motivating **`LFI-S2` (`C2 @ H/2` skip)**.

---

## 3. Gap 2 & Gap 4 Proof — Dense Undercounting, Gradient Decay & Background Fog Losses

- **Bayesian Loss (`BL+`)** (Ma et al., *ICCV 2019*, [ar5iv:1908.03684](https://ar5iv.labs.arxiv.org/html/1908.03684)): Replaces Gaussian targets with posterior expectation $\mathbb{E}[c_n] \to 1$ per head and dummy background expectation $\mathbb{E}[c_0] \to 0$ to suppress background fog.
- **DM-Count** (Wang et al., *NeurIPS 2020*, [ar5iv:2009.13077](https://ar5iv.labs.arxiv.org/html/2009.13077)): Proves Gaussian smoothing loosens generalization bounds and KL/JS fails on disjoint supports; combines `L1 Count + Sinkhorn OT + TV`.
- **Generalized Loss (`GL`)** (Wan et al., *CVPR 2021*, [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wan_A_Generalized_Loss_Function_for_Crowd_Counting_and_Localization_CVPR_2021_paper.html)): Formulates density learning as unbalanced OT and proves pixel L2 and BL are suboptimal special cases.
- **ChfL** (Shu et al., *CVPR 2022*, [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Shu_Crowd_Counting_in_the_Frequency_Domain_CVPR_2022_paper.html)): Proves characteristic-function frequency distance upper-bounds the **pseudo sup-norm error across all spatial sub-regions**, eliminating Gaussian-overlap saturation in dense clusters.
- **CLIP-EBC** (Ma et al., *arXiv:2403.09281*, 2024, [HTML](https://arxiv.org/html/2403.09281)) & **CPMS-Net (`RCJL`)** ([DOI](https://doi.org/10.1007/s44443-026-01052-3)): Prove pure global/classification objectives undercount dense tails; CLIP-EBC merges dense bins `[m, \infty)` + discrete Wasserstein count (`DACE`), while `CPMS-Net` adds `4x4` non-overlapping grid `L1` count conservation (`RCJL`).
- **Connection to Ours:** Explains why `count_loss_mode: nb` (`r=50`, `1/y` gradient decay) starves `3000+` scenes while normalized `Radon-OT` carries zero count gradient—motivating `D0.5` (unbounded `linear` HF) $\to$ `D1` (`Anscombe + L1 0.02`) and optionally `4x4 Grid-L1` (`RCJL`, `0` inference params) if dense clusters still lag.

---

## 4. Normalization (`BN` vs `GN`/`SGN`) & Real-Time Edge Folding Tradeoff

- **Group Normalization** (Wu & He, *ECCV 2018*, [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)), **SACANet** (Bai et al., [ar5iv:1909.03839](https://ar5iv.labs.arxiv.org/html/1909.03839)), & **Lw-Count** (Liu et al., *TCSVT 2022*): Prove small-batch high-resolution crowd training with extreme empty-to-dense patch variance corrupts BN running stats; GN / Spatial GroupNorm (SGN) stabilizes convergence. **MDKNet** ([arXiv:2402.03758](https://arxiv.org/html/2402.03758)) similarly shows cross-density mixing biases shared BN statistics.
- **RepVGG** (Ding et al., *CVPR 2021*, [arXiv:2101.03697](https://arxiv.org/abs/2101.03697)): Proves only static-stat `BatchNorm` algebraically folds into `Conv2d` at zero inference latency, whereas per-sample `GN`/`SGN` cannot fold.
- **Connection to Ours:** Directly verified in our ORT CPU 1-thread benchmark (`results/jrtip_realtime_profile_lfi_s2_cpu1t_onnx.json`): `GroupNorm` at Stride 2 crushed `LFI-S2` to `13.1 FPS` (`512`) / `1.4 FPS` (`1080p`), while replacing Stride-2 `GN` with foldable `BatchNorm2d` (keeping `GN` at coarse `S4/8/16`) doubled throughput to **`25.9 FPS` / `2.7 FPS`**.

---

## 5. 4K Ultra-Dense Inference Protocols (`2048` Cap vs `3072` / Sliding-Window Tiling)

- **Standard `<=2048` Protocol:** **NWPU-Crowd** ([ar5iv:2001.03360](https://ar5iv.labs.arxiv.org/html/2001.03360)), **JHU-Crowd++** ([ar5iv:2004.03597](https://ar5iv.labs.arxiv.org/html/2004.03597)), **FIDT** (`<2048`), **DM-Count** (`NWPU 1920`), **P2PNet** (`NWPU 1920`), and **Zhao et al.** (`512x1024`) downscale large scenes for single-pass memory/speed.
- **Explicit Literature Precedent That `2048` Breaks `3000+` Ultra-Dense Scenes:**
  - **ChfL** (*CVPR 2022*, [preprocessor.py](https://github.com/wbshu/Crowd_Counting_in_the_Frequency_Domain/blob/main/Dataset/preprocessor.py)) caps NWPU Val at `2048` **except explicitly hardcoding `exception=(3234,)` resized to `3024`** because `2048` destroys head spacing on NWPU image `3234`!
  - **CLIP-EBC** ([preprocess.sh](https://github.com/Yiming-M/CLIP-EBC/blob/main/preprocess.sh), [test_nwpu.sh](https://github.com/Yiming-M/CLIP-EBC/blob/main/test_nwpu.sh)) raises NWPU cap to **`max_size=3072`** and evaluates with **sliding-window tiling** (`58.2` MAE).
- **Connection to Ours:** Our worst outlier on NWPU (`99,766`-param `LFI-99k`) is literally **`id=3234.0`** (`GT 12,924 -> Pred 4,272.8`, `-8,651.2`), matching ChfL's exact hardcoded exception! Run **`D3` (`eval_nwpu_tiling_sweep.py`)** on existing `best_val_mae.pt` to quantify input downscale loss vs model capacity directly.
