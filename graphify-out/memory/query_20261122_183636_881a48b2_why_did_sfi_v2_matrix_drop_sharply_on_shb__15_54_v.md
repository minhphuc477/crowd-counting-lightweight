---
type: "query"
date: "2026-11-22T18:36:36.855753+00:00"
question: "Why did sfi_v2_matrix drop sharply on SHB (15.54 vs 9.53), NWPU (87.72 vs 81.17), and JHU (61.17 vs 58.77)?"
contributor: "graphify"
outcome: "corrected"
correction: "Use invariant nextgen_sub65 optimizer hyperparameters (lr=0.0003, backbone_lr_scale=0.25, pad_small_images=True, max_size=2048) across all benchmark comparisons."
source_nodes: ["RMRv3Config", "SFILiteFPNNeck"]
---

# Q: Why did sfi_v2_matrix drop sharply on SHB (15.54 vs 9.53), NWPU (87.72 vs 81.17), and JHU (61.17 vs 58.77)?

## Answer

configs/benchmarks/sfi_v2_matrix/*.yaml silently lowered train.lr from 0.0003 to 0.0001 and backbone_lr_scale from 0.25 to 0.1 (7.5x smaller backbone LR), plus disabled pad_small_images and max_size on SHB.

## Outcome

- Signal: corrected
- Correction: Use invariant nextgen_sub65 optimizer hyperparameters (lr=0.0003, backbone_lr_scale=0.25, pad_small_images=True, max_size=2048) across all benchmark comparisons.

## Source Nodes

- RMRv3Config
- SFILiteFPNNeck