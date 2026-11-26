# LITERATURE_REVIEW.md — Snowballing Literature Map & Gap Verification

Systematic backward & forward snowballing across Semantic Scholar and arXiv (2018–2026). Every metric below was directly verified on the source page/PDF (`Not verified on page` = behind publisher paywall).

---

## 1. Snowballing Search Log & Seeds
- **Platforms:** Semantic Scholar (`semanticscholar.org`), arXiv (`arxiv.org`).
- **Queries:** `"lightweight crowd counting"`, `"Compact Convolutional Neural Network" / "C-CNN"`, `"GhostNet crowd counting"`, `"Frequency-Aware Feature Fusion" FreqFusion`, `"Laplacian feature pyramid high-frequency"`, `"FADE upsampling"`, `"HS-FPN"`, `"AFPN"`, `"BiFPN"`, `"Slim-Neck GSConv"`, `"Gold-YOLO"`.
- **Backward/Forward Seeds:**
  - *Crowd Counting:* P2PNet (ICCV 2021), PET (ICCV 2023), STEERER (ICCV 2023), MobileCount (PRCV 2019 / Neurocomputing 2020), SKT (ACM MM 2020), TinyCount (JRTIP 2024), LMSFFNet (TGRS 2023).
  - *Frequency / Laplacian FPN:* CARAFE (ICCV 2019), DHPF (ICCV 2021) $\to$ FADE (ECCV 2022 / IJCV 2024), LFPN (TGRS 2022) $\to$ FreqFusion (TPAMI 2024), HS-FPN (AAAI 2025) $\to$ S2FCNet (TGRS 2025), SFM (TPAMI 2025), LaFINet (AAAI 2026), HF-DETR (IEEE SPL 2026), FSDC-DETR (ECCV 2026).

---

## 2. Lightweight Crowd Counting Competitors (<300k Params + Key Seeds)

| Method | Venue / Year | Params | FLOPs | KD | SHA (MAE/RMSE) | SHB (MAE/RMSE) | QNRF (MAE/RMSE) | NWPU (MAE/RMSE) | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LCN** | *ICIP* 2019 | `32k` | — | — | `93.3 / 157.0` | `15.1 / 23.3` | `262.0 / 358.6` | — | [LSKD PDF](https://pdfs.semanticscholar.org/306d/86451fb5f7e5ac1489605f4d7f2d497ca895.pdf) |
| **LCDnet** | *arXiv* 2023 | `50k` | `4.85G` | No | *Not evaluated on SHA* *(DroneRGBT 21.4, CARPK 13.1)* | — | — | — | [arXiv:2302.05374](https://arxiv.org/abs/2302.05374) |
| **TinyCount** | *JRTIP* 2024 | `60k` | — | No | *Paywall* | *Paywall* | *Paywall* | — | [S2](https://www.semanticscholar.org/paper/TinyCount%3A-an-efficient-crowd-counting-network-for-Lee-Lee/cae1b7e6de2540536c3fb5664fc148f2e8f0e25f) |
| **C-CNN** | *ICASSP* 2020 | `70k–73k` | `19.83G` | No | `88.1 / 141.7` | `14.9 / 22.1` | — | — | [arXiv:2002.06515](https://arxiv.org/abs/2002.06515) |
| **Lw-Count** | *TCSVT* 2022 | `71k` | — | No | `69.7 / 100.5` | `10.1 / 12.4` | `149.7 / 238.4` | — | [S2](https://www.semanticscholar.org/paper/Lw-Count%3A-An-Effective-Lightweight-Crowd-Counting-Liu-Cao/6528f558a266116ae169c6eb2720e5a99adf6882) |
| **EdgeCount** | *TCSVT* 2025 | `120k` | `19.87M*` | **Yes** | *Paywall* | *Paywall* | *Paywall* | — | [S2](https://www.semanticscholar.org/paper/A-Lightweight-Object-Counting-Network-Based-on-Map-Shen-Li/8c0d9c692b0ad88f3a9f620fd3805c290b4053b3) |
| **CPMS-Net** | *JKSUCIS* 2026 | `120k` | — | No | *Paywall* | *Paywall* | `126.7 / 201.0` | — | [S2](https://www.semanticscholar.org/paper/CPMS-Net%3A-A-lightweight-cooperative-perception-for-Shang-Weng/9ae103811d32727b08b00c4d464f83cc8c5f1a6f) |
| **Zhao et al.** | *arXiv* 2025 | `149k` | `1.32G` | No | `63.4 / 106.3` | `7.3 / 11.2` | `104.1 / 172.2` | `102.5 / 388.6` | [arXiv:2510.13250](https://arxiv.org/abs/2510.13250) |
| **LSANet** | *Neurocomp.* 2022 | `200k` | `5.41G` | No | `66.1 / 110.2` | `8.6 / 13.9` | `112.3 / 186.9` | — | [S2](https://www.semanticscholar.org/paper/Real-time-crowd-counting-via-lightweight-network-Zhu-Yan/c5db3ac46aef75d2ddb8971b87582b8d4b23e966) |
| **STRmt** | *CCPE* 2024 | `240k` | `2.04G` | — | `64.7 / 111.8` | `7.6 / 11.8` | — | `120.1 / 435.8` | [S2](https://www.semanticscholar.org/paper/STRmt%3A-A-state-transition-based-model-for-real%E2%80%90time-Sun-Zhao/54369000be38416f1fecfef0cc2768782fc8ac28) |
| **1/5-CSRNet+SKT** | *ACM MM* 2020 | `640k` | `8.45G` | **Yes** | `RMSE 118.78` | — | — | — | [S2](https://www.semanticscholar.org/paper/Efficient-Crowd-Counting-via-Structured-Knowledge-Liu-Chen/d0f5b5d2592d5802a6181860fe5f132dc83526ae) |
| **MobileCount** | *Neurocomp.* 2020 | `3.40M` | `6.15G` | No | `89.4 / 146.0` | `9.0 / 15.4` | `131.1 / 222.6` | — | [S2](https://www.semanticscholar.org/paper/MobileCount%3A-An-Efficient-Encoder-Decoder-Framework-Wang-Gao/82f18b696cbe2290e536ecf4e22620e7299691b9) |

*Additional related 2022–2026 lightweight counting works (paywalled metrics):* **PDDNet** (*Applied Intelligence* 2022), **GAPNet** (*FGCS* 2023), **LMSFFNet** (*IEEE TGRS* 2023), **LEDCrowdNet** (*IEEE IoT-J* 2024), **LMGNet** (*IEEE TIM* 2025), **PEGNet** (*IEEE TITS* 2026).

---

## 3. Frequency / Laplacian FPN & Lightweight Small-Object Necks

| Paper | Venue / Year | Mechanism & Comparison to `LFI-Lite` |
| :--- | :--- | :--- |
| **CARAFE** / **DHPF** | *ICCV* 2019 / 2021 | Backward seeds: content-aware upsampling / dynamic high-pass filtering for SR; no FPN AvgPool subtraction. |
| **FADE** ([arXiv:2407.13500](https://arxiv.org/abs/2407.13500)) | *ECCV* 2022 / *IJCV* 2024 | Task-agnostic encoder-decoder upsampling + gating; not sub-100k crowd counting. |
| **LFPN** ([S2](https://www.semanticscholar.org/paper/Laplacian-Feature-Pyramid-Network-for-Object-in-VHR-Zhang-Jiao/3de30852e4675dc4a6205285b22ffcb300831aa7)) | *IEEE TGRS* 2022 | Parallel Laplacian pyramid pathway for VHR remote sensing detection; not sub-100k crowd counting. |
| **FreqFusion** ([arXiv:2408.12879](https://arxiv.org/abs/2408.12879)) | *IEEE TPAMI* 2024 | ALPF + offset resampling + AHPF (identity minus **Softmax-normalized** dynamic $3\times 3$ kernel); not norm-free linear AvgPool, not sub-100k crowd counting. |
| **HS-FPN** ([arXiv:2412.10116](https://arxiv.org/abs/2412.10116)) | *AAAI* 2025 | **DCT** high-pass filter generating channel/spatial attention masks + cross-attention (`71.73M` w/ R-50) for tiny object detection. |
| **SFM** / **S2FCNet** | *TPAMI* 2025 / *TGRS* 2025 | Forward from FreqFusion: adaptive resampling modulation / semantic-spatial compensation for segmentation & remote-sensing TOD. |
| **LaFINet** / **HF-DETR** / **FSDC-DETR** | *AAAI* 2026 / *SPL* 2026 / *ECCV* 2026 | Camouflaged detection Laplacian injection / `8.6M` LoG+wavelet UAV-DETR / DETR frequency-spatial encoder. |
| **PAFPN / BiFPN / Slim-Neck / AFPN / Gold-YOLO / IHP+SNI** | *CVPR* 2018–*ECCV* 2024 | Canonical lightweight & small-object necks ([PAFPN](https://arxiv.org/abs/1803.01534), [BiFPN](https://arxiv.org/abs/1911.09070), [Slim-Neck](https://arxiv.org/abs/2206.02424), [AFPN](https://arxiv.org/abs/2306.15988), [Gold-YOLO](https://arxiv.org/abs/2309.11331), [IHP+SNI](https://arxiv.org/abs/2505.12820)); none use norm-free linear Laplacian HF residual at `P4/L4`. |

---

## 4. Gap Check & Literature Foundations
- **Gap Verified (`0` prior matches):** No found paper combines **(1) sub-100k KD-free crowd counting** with **(2) a norm-free linear Laplacian HF residual on FPN `P4/L4` modulated by a WD-exempt channel $\gamma$**.
- **Point Supervision & Resolution:** **Bayesian Loss** (*ICCV* 2019, [ar5iv:1908.03684](https://ar5iv.labs.arxiv.org/html/1908.03684)), **P2PNet** (*ICCV* 2021), **STEERER** (*ICCV* 2023), **ChfL** (*CVPR* 2022, NWPU `id=3234` `max_size` exception), **CLIP-EBC** (`2024`, `max_size=3072` tiling).
- **Normalization & Folding:** **GroupNorm** (*ECCV* 2018) stabilizes small-batch crowd variance; **RepVGG** (*CVPR* 2021) folds `BatchNorm` into `Conv2d` for CPU latency.
