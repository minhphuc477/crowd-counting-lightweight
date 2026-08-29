# NTPC CVPR-readiness audit

Audit date: 2026-08-30  
Branch: `feat/ntpc-neural-tree-polya`

## Verdict

The implementation and official ShanghaiTech installation are coherent enough for the one-seed formulation study, but the architecture is not yet empirically proven to be “ultimate” or CVPR-ready. The deployed graph is genuinely lightweight; the accuracy, novelty strength, and Pareto superiority still require real results and causal ablations.

## Official ShanghaiTech provenance and annotation policy

- The dataset was downloaded from the 2024 link in Desen Zhou's official repository. Archive size is 174,398,901 bytes and SHA-256 is `95B7A7F3D927F756393005B98F77847CE40DCB8546E2F46D20F0E0D425DAC403`.
- All 2,396 current files are byte-identical to that archive; split counts are SHA `300/182` and SHB `400/316`.
- The official archive itself contains source annotations outside paired image bounds (including large outliers), so this is not a local download mismatch.
- PET and DM-Count official loaders preserve raw ShanghaiTech coordinates, filter points when sampling a training crop, and use original cardinality at evaluation. NTPC now follows that convention for SHA/SHB only. It does not clamp or silently change the official count; QNRF/NWPU/custom validation remains strict.

## Verified implementation properties

- One positive stride-4 mass map is conserved exactly by sum pooling through the count tree.
- R4 implements the intended `N -> 64 -> 32 -> 16` factorization; R5 adds only the dense-parent `16 -> 8` auxiliary.
- The root Negative-Binomial and conditional Dirichlet-Multinomial terms are evaluated in FP32 with finite/target-conservation checks.
- The deployed model has 96,593 parameters and 0.075764 Conv2d GMAC at `256x256` (interpolation, normalization, activation, and OT-M costs are excluded from this Conv-only figure).
- On the audit machine (RTX 3050 Ti Laptop GPU, batch 1, `256x256`) the physically C16-truncated graph measured 6.93 ms median, 8.99 ms p95, and 5.45 MiB peak allocated memory. These numbers are machine-specific, not cross-paper claims.
- Parameter-free local-max and OT-M localization remain offline secondary decoders; neither changes the counting graph.
- Full-image inference remains the official default. An explicit stride-aligned tiled mode is available for very large images; because GroupNorm statistics are tile-local, tiled results are a separately recorded protocol rather than a silent OOM fallback.

## Bugs/remediation completed

- Enabled a pinned timm ImageNet-1k initialization for every matched R0--R5/depth config.
- Corrected the pretrained input contract from legacy ImageNet mean/std to the weight metadata `(0.5, 0.5, 0.5)`.
- Added fail-fast normalization/source validation and checkpoint provenance.
- Added disjoint AdamW parameter groups with a configurable backbone LR scale (`0.1` in the first study).
- Prevented evaluate/profile/export/localization tools from downloading pretrained weights that a task checkpoint immediately overwrites.
- Added backbone/task LR logging plus scheduler state to checkpoints.
- Fixed `create_smoke_dataset.py --help`, which previously created files as a side effect; it now has a real CLI and refuses non-empty output directories.
- Removed import-time execution from `summary_runs.py` and added a proper CLI.
- Corrected `requirements-lock.txt` to the environment actually used for this audit.
- Centralized SHA/SHB/QNRF/NWPU evaluation construction so counting, localization, and visualization cannot silently choose different splits or coordinate conventions.
- Replaced component-wise dense Hungarian allocation with exact maximum-cardinality matching on a sparse distance-gated graph; a 10,000-point connected-component regression test prevents dense-matrix regressions.
- Collapsed equal-valued local-maximum plateaus to one deterministic point.
- Added explicit OT-M full-resolution initialization limits and a recorded `stride_grid` alternative for large images; Oracle-cardinality evaluation is now recorded in both metadata and per-image rows.
- Added deterministic generator plumbing to the weighted sampler, positive `pad_multiple` validation, one-sync gradient diagnostics, and tests that data-driven head initialization cannot alter backbone weights.
- Locked the causal ablation contract: R0--R5/T1/T2 may differ only in experiment identity and loss; dataset, model, augmentation, statistics, sampler, optimizer, schedule, and training sections are tested equal.
- Corrected timm MobileNetV4 feature extraction: `MobileNetV3Features` returns selected tensors but still iterates all stages, so NTPC now physically removes `blocks.3` and `blocks.4` after loading pretrained weights. Retained pretrained tensors, C4/C8/C16 outputs, the complete NTPC mass/count output, and all retained gradients are bit-exact to the untruncated graph; discarded parameters are verified to have `grad=None` in the full graph.
- Final project validation: `87 passed`; Python byte-compilation, evaluator CLI checks, and launcher dry-run complete without code errors. The shared Python environment still has unrelated optional-package conflicts reported by `pip check` (for example MLflow/Unsloth); the NTPC runtime versions are pinned separately in `requirements-lock.txt`.

## Evidence still required before a CVPR claim

1. Run the requested one-seed R0--R5 study and reject the core hypothesis unless R4 beats the deterministic R1 and flat-DM R2 controls under the identical protocol.
2. Compare R4, R5, DTM8, and DTM4 for both counting and localization. Deeper supervision is not a contribution unless it improves the measured trade-off.
3. Add a pretrained-vs-scratch control and a backbone LR-scale sensitivity check on the finalist. Pretraining is now the default protocol, not evidence that `0.1` is optimal.
4. Report SHA, SHB, QNRF, and NWPU accuracy together with parameters, full-graph MAC/FLOP methodology, measured latency, and memory on fixed hardware/software.
5. Compare against strong recent counting/localization methods using their published protocols. The architectural pieces (MobileNet, FPN, Softplus head) are not individually novel; the defensible novelty is the conserved Tree-Pólya allocation objective and its demonstrated behavior.
6. After selecting the formulation with one seed, use multiple confirmation seeds for the final paper tables. This is intentionally deferred during the current exploratory phase.

## Decision boundary

Do not claim “ultimate architecture” from parameter count or unit tests. A defensible claim is narrower: **a sub-0.5M conserved-mass counter whose hierarchical probabilistic objective improves matched baselines and optionally yields useful parameter-free localization**. If R4 does not beat R1/R2, the Tree-Pólya contribution is falsified in its current form and the architecture should not be cosmetically expanded to rescue the claim.
