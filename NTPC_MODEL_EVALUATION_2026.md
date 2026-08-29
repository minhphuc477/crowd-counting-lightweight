# NTPC model evaluation against the 2023--2026 landscape

Date: 2026-08-30  
Branch: `feat/ntpc-neural-tree-polya`  
Evidence status: implementation verified; current benchmark checkpoint/results unavailable

## Executive verdict

NTPC is a credible ultra-lightweight research prototype, not yet a demonstrated CVPR model. Its strongest contribution is the exact conserved factorization

`root count -> regional allocation -> recursive child allocation`

with Negative-Binomial and Dirichlet-Multinomial likelihoods over one positive stride-4 mass field. The architecture itself is intentionally conventional: 340,992 of 350,017 parameters (97.4%) are in the pretrained MobileNetV4 backbone. Therefore the paper should claim a **probabilistic hierarchical training formulation and conserved representation**, not a fundamentally new backbone.

Current recommendation: **promising but empirically incomplete / weak reject if submitted now**. The recommendation can move to borderline accept or accept only after the causal study and cross-dataset results validate the central claim.

## Evidence reviewed

Exa Search reviewed 106 search hits across three workstreams: ultra-lightweight counting, counting/localization methodology, and efficiency/reproducibility. Eight primary papers/repositories and the official CVPR 2026 review criteria were deep-read. Search results were treated as candidates, not evidence; the assessment below relies on primary papers, official proceedings, and official repositories.

## Quantitative implementation assessment

| Property | Verified NTPC value | Interpretation |
|---|---:|---|
| Parameters | 350,017 | Genuinely sub-0.5M |
| Backbone parameters | 340,992 (97.4%) | Novelty is primarily the objective, not the feature extractor |
| Neck + head | 9,025 (2.6%) | Very small task-specific graph |
| Conv MACs, 256x256 | 0.09421 GMAC | Verified by forward hooks |
| Conv MACs, 1920x1080 | 2.98938 GMAC | Measured at the comparison resolution |
| Conv FLOPs, MAC=2 ops | 5.97876 GFLOPs | Convolution only; GN/interpolation/activations excluded |
| 1920x1080 latency | 17.77 ms median, 18.17 ms p95 | 56.3 FPS, batch 1, FP32, RTX 3050 Ti Laptop |
| 1920x1080 peak allocation | 161.36 MiB | PyTorch CUDA allocator, inference without profiling hooks |
| Automated verification | 84 tests passed | Includes conservation, gradients, 10k localization matching, OT-M and tiled inference |

The nearest efficiency competitor found is ZIP-P, which also uses a MobileNetV4-Small 0.5x family backbone and probabilistic discrete-count modeling. ZIP-P reports 0.81M parameters and 6.46 GFLOPs at 1920x1080, with MAE 71.18/8.23/96.29 on SHA/SHB/QNRF. FLOP definitions must be re-measured in the same profiler before claiming NTPC is cheaper, but NTPC already uses approximately 57% fewer parameters.

## Reviewer-style scorecard

| Dimension | Score | Evidence-based assessment |
|---|---:|---|
| Technical soundness | 8/10 | Conservation and the hierarchical likelihood are internally coherent and tested |
| Architectural efficiency | 9/10 | 0.35M parameters, practical 1080p memory and latency |
| Method novelty | 6.5/10 | Tree-structured DM allocation is interesting, but ZIP already occupies probabilistic lightweight counting and PML formalizes modern density losses |
| Experimental design | 7.5/10 | R1/R2/R3/R4 isolate deterministic, flat, multinomial, and overdispersed hierarchy unusually well |
| Empirical evidence | 1/10 | No current compatible checkpoint or verified MAE/RMSE table |
| Localization contribution | 5/10 | Useful parameter-free secondary capability, but OT-M is prior work and its compute must be reported separately |
| Reproducibility | 8/10 | Pinned configs/data hash/seed/weights/tests; final multi-seed confirmation and clean environment remain |
| Current CVPR readiness | 4/10 | Strong engineering and hypothesis, insufficient evidence |

## Closest prior work and consequence for positioning

| Work | Relevant overlap | What NTPC must demonstrate |
|---|---|---|
| ZIP (2025 preprint) | Probabilistic non-negative block counts, MobileNetV4, lightweight scaling | Hierarchical DM allocation adds value beyond zero-aware Poisson modeling |
| PET (ICCV 2023) | Quadtree decomposition; joint counting/localization | NTPC is much smaller, but localization is post-processing rather than learned point querying |
| OT-M (CVPR 2023) | Parameter-free density-to-point localization | Present OT-M as an external decoder, not an NTPC architectural invention |
| PML (ICLR 2025) | Principled density loss and counting/localization unification | Explain why conserved hierarchical count likelihood is different and when it wins |
| P2R (CVPR 2025) | Point-to-region supervision for localization/counting | Do not claim point supervision or semi-supervision unless implemented and evaluated |
| TinyCount (2024) | Extreme parameter minimization, reported around 0.06M | NTPC needs substantially better accuracy to justify six times more parameters |

## Strongest aspects

1. **Exact conservation:** every reported regional count and the global count comes from sums of the same positive mass field; there are no inconsistent auxiliary heads.
2. **Clean causal ablations:** R3 versus R4 directly tests multinomial versus Dirichlet-Multinomial overdispersion; R2 versus R4 tests flat versus hierarchical allocation; R1 versus R4 tests deterministic versus probabilistic allocation.
3. **Deployment credibility:** the task-specific graph is only 9,025 parameters, full-HD inference fits comfortably on the audit GPU, and tiled inference is explicit rather than silently changing protocol.
4. **Secondary localization:** OT-M can test whether deeper conserved supervision makes the mass representation more instance-aware without adding trained parameters.

## Main rejection risks

1. **No empirical evidence yet.** Unit tests establish correctness, not crowd-counting quality.
2. **Novelty collision with ZIP.** Both projects use probabilistic discrete counts and MobileNetV4. The paper needs a direct same-backbone comparison or a faithful ZIP-like baseline.
3. **The word “architecture” is too strong.** Almost all parameters come from an existing backbone; the contribution is principally the loss/factorization.
4. **Fixed dispersion/concentration.** Root dispersion and level-wise kappa values are fixed. Reviewer questions about calibration and sensitivity are unavoidable unless likelihood/NLL and sensitivity results are reported.
5. **Test-set model selection.** It matches common SHA code practice and is controlled across ablations, but it is not an unbiased held-out estimate. Results should additionally include a frozen-epoch or fixed-schedule report and label the selection protocol precisely.
6. **OT-M is not free computationally.** “Zero learned parameters” must not be written as “zero cost”; post-processing latency, memory, initialization mode, and cardinality source must be disclosed.
7. **Efficiency comparison can be misleading.** Parameters, MACs, latency, memory, and resolution must all be reported; FLOPs from different libraries are not directly comparable.

## One-seed falsification plan

Run the already locked seed-42 order first. Do not add architectural modules before these gates are evaluated.

### Gate 1: Does hierarchy help?

- R4 must beat R2 on MAE and not materially regress RMSE.
- R4 must beat R1; otherwise the gain is allocation supervision, not Tree-Polya uncertainty.
- R4 must beat R3; otherwise DM overdispersion is unnecessary and multinomial hierarchy is the simpler method.

Failure of any comparison is informative. Failure of both R4>R2 and R4>R3 falsifies the current central claim.

### Gate 2: Does adaptive depth help?

- Compare R4, R5, T1/DTM8, and T2/DTM4 for MAE/RMSE and localization F1@4/F1@8.
- Keep R5 only if its dense-parent branch improves the Pareto trade-off.
- Keep DTM4 only if localization improves without a meaningful counting regression.

### Gate 3: Is it competitive for its size?

As an initial ultra-lightweight target, results should at least approach the published sub-1M envelope: approximately SHA MAE 63--71, SHB MAE 8--9, and QNRF MAE 96--112. These are landscape ranges, not acceptance thresholds. A compelling paper should either reach the stronger end of the range or show a clear new insight with a better efficiency/robustness trade-off.

### Gate 4: Can the probabilistic claim be defended?

Report test NLL under a frozen configuration, count bias, NAE, dense/sparse subgroup errors, and sensitivity to root dispersion and per-level kappa. Add a same-architecture Poisson or ZIP-like control. If likelihood gains do not translate to MAE, calibration, robustness, or localization, the probabilistic story is not supported.

## Minimum final-paper package

- One-seed formulation selection, followed by at least three confirmation seeds for the finalist and strongest baseline.
- SHA, SHB, QNRF, and NWPU results; UCF-CC50 only with official five-fold evaluation.
- Same-resolution Params/MACs/FLOPs/latency/throughput/peak-memory table with hardware, precision, batch size, warmup, and software versions.
- R0--R5/T1/T2 causal table plus pretrained-versus-scratch and LR-scale sensitivity.
- Native count versus Oracle-cardinality localization clearly separated.
- Full-image and tiled protocols never mixed in one headline table.
- Limitation statement: reliance on pretrained MobileNetV4, fixed distributional hyperparameters, and OT-M post-processing cost.

## Primary sources

- CVPR 2026 reviewer criteria: https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines
- ZIP: https://arxiv.org/html/2506.19955v3
- PET: https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Point-Query_Quadtree_for_Crowd_Counting_Localization_and_More_ICCV_2023_paper.html
- OT-M: https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.html
- PML: https://proceedings.iclr.cc/paper_files/paper/2025/hash/04c956c52bfc1cc5f6dd989c729213b7-Abstract-Conference.html
- P2R: https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Point-to-Region_Loss_for_Semi-Supervised_Point-Based_Crowd_Counting_CVPR_2025_paper.html
- TinyCount official repository: https://github.com/HBL03/TinyCount
- Efficiency Pentathlon: https://arxiv.org/abs/2307.09701
