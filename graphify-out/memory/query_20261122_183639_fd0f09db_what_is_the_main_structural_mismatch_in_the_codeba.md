---
type: "query"
date: "2026-11-22T18:36:39.775179+00:00"
question: "What is the main structural mismatch in the codebase identified by Graphify?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["RMRv3", "build_multiscale_regions", "unrolled_sirt_solver", "SFILiteFPNNeck", "FineMeasureHead"]
---

# Q: What is the main structural mismatch in the codebase identified by Graphify?

## Answer

Top God Nodes are still legacy solver/region code (RMRv3 441 edges, build_multiscale_regions 261 edges, regional_sum 139 edges, unrolled_sirt_solver 86 edges) plus a train.py<->trainer.py import cycle, even though active best models run enable_solver=False with SFILiteFPNNeck + FineMeasureHead.

## Outcome

- Signal: useful

## Source Nodes

- RMRv3
- build_multiscale_regions
- unrolled_sirt_solver
- SFILiteFPNNeck
- FineMeasureHead