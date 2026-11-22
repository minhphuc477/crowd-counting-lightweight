---
type: "query"
date: "2026-11-22T19:02:04.886291+00:00"
question: "Why did gate4 collapse around 0.50 in SFILiteFPNNeck and why did axis_a_v2_no_gate beat V4 on SHA?"
contributor: "graphify"
outcome: "corrected"
correction: "Decouple coarse-context gating from high-frequency modulation in SFILiteFPNNeck (e.g., complementary gate4*up8_to_4 + (1-gate4)*hf_raw or separate gates) before running new benchmarks."
source_nodes: ["SFILiteFPNNeck"]
---

# Q: Why did gate4 collapse around 0.50 in SFILiteFPNNeck and why did axis_a_v2_no_gate beat V4 on SHA?

## Answer

Code inspection of rmr_core/necks/sfi.py (L280 and L304) shows SFILiteFPNNeck multiplies BOTH coarse context up8_to_4 (L280: l4 + gate4 * up8_to_4) AND high-frequency impulse hf_raw (L304: gamma_hf * gate4 * hf_raw) by the exact same gate4. This creates a mathematical tug-of-war: boosting HF requires gate4->1 while suppressing coarse smearing requires gate4->0, pinning gate4 near 0.50 (0.4958 bg vs 0.5079 dense on SHA) and degrading V4 Full SFI-v2 (67.35) vs V2 No Gate (64.05).

## Outcome

- Signal: corrected
- Correction: Decouple coarse-context gating from high-frequency modulation in SFILiteFPNNeck (e.g., complementary gate4*up8_to_4 + (1-gate4)*hf_raw or separate gates) before running new benchmarks.

## Source Nodes

- SFILiteFPNNeck