---
type: "query"
date: "2026-11-22T19:02:05.291954+00:00"
question: "What is the single invariant hyperparameter baseline for clean nextgen_sub65 runs?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["SFILiteFPNNeck", "FineMeasureHead", "RMRv3Config"]
---

# Q: What is the single invariant hyperparameter baseline for clean nextgen_sub65 runs?

## Answer

Remaining verified completed runs use lr=0.0003, backbone_lr_scale=0.25, crop_size=512, max_size=2048, pad_small_images=True under nextgen_sub65, yielding 81.17 MAE (4.60 zero-count error) on NWPU with SFI-v2+DSIMR and 63.25 MAE on SHA / 99.04 MAE on QNRF / 57.45 JHU Val with sfi_lite+dw_subpixel.

## Outcome

- Signal: useful

## Source Nodes

- SFILiteFPNNeck
- FineMeasureHead
- RMRv3Config