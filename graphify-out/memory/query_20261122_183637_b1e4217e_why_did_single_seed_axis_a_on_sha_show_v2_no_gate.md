---
type: "query"
date: "2026-11-22T18:36:37.549507+00:00"
question: "Why did single-seed Axis A on SHA show V2 No Gate (64.05) and Frankenstein (63.92) beating V4 Full SFI-v2 (67.35)?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["SFILiteFPNNeck", "FrankensteinSOTALiteNeck", "FineMeasureHead"]
---

# Q: Why did single-seed Axis A on SHA show V2 No Gate (64.05) and Frankenstein (63.92) beating V4 Full SFI-v2 (67.35)?

## Answer

SHA has only 182 test images and sfi_lite 3-seed variance is 65.38 +/- 3.32 MAE (63.25, 63.68, 69.21). Single-seed diffs within 3.4 MAE on SHA fall inside stochastic seed noise, whereas Frankenstein gate flatlined (Delta=-0.0002) and ran 19-40% slower (394 FPS vs 469-552 FPS).

## Outcome

- Signal: useful

## Source Nodes

- SFILiteFPNNeck
- FrankensteinSOTALiteNeck
- FineMeasureHead