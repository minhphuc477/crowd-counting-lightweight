---
type: "query"
date: "2026-11-22T18:36:38.441169+00:00"
question: "Where does SFI-v2 + DSIMR provide unambiguous architectural gain?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["SFILiteFPNNeck", "FineMeasureHead"]
---

# Q: Where does SFI-v2 + DSIMR provide unambiguous architectural gain?

## Answer

On multi-regime NWPU-Crowd (500 val), exp_nwpu_sfi_v2_dsimr_subpixel_101k reached 81.17 MAE (vs 94.92 baseline, 86.30 bg_ratio010) and slashed N=0 negative-scene phantom error from 34.64/68.01 down to 4.60, with gate response Delta=+0.0504 and dsimr_proj bias=[-2.69, -7.28].

## Outcome

- Signal: useful

## Source Nodes

- SFILiteFPNNeck
- FineMeasureHead