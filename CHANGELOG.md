# Changelog

All notable changes to the RMR (Radon Measure Recovery) codebase are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [5.0.0] - 2026-10-18

### Added
- **RepASPPLiteNeck (`rmr_core/necks/rep_aspp.py`)**:
  - Multi-scale FPN neck fusing ASPP-Lite on $L_{16}$ (dilations $\{1, 3, 6\}$ + Global Average Pooling linear projection for $512\text{ px}$ macro-crowd context) with structural re-parameterizable $7\times 7$ depthwise blocks (`RepDWBlock7x7`) across all pyramid levels ($P_{16}, P_8, P_4$).
  - Zero-latency deployment inference via `switch_to_deploy()` algebraic fusion into single $7\times 7$ depthwise convs.
  - Fits within parameter budgets: Track A has **104,359 params** ($\le 104,441$, buffer = 82 params); Track B has **125,786 params** ($\le 200,000$, buffer = 74,214 params).
- **Track A Configuration (`configs/rmr_v50/v50_track_a_repaspp.yaml`)**:
  - Full single-variable standalone configuration for ShanghaiTech Part A ($512\times 512$ crop, 0.0% Knowledge Distillation).
  - Native multi-scale pooling ($32\text{ px} \to P_4, 64\text{ px} \to P_8, 128\text{ px} \to P_{16}$).
  - Purified orthogonal 3-loss suite with Seitzer et al. (ICLR 2022) Beta-NB NLL.
- **Track B Configuration (`configs/rmr_v50/v50_track_b_nextgen.yaml`)**:
  - Extended feature width $W=64$ with Subpixel Stride 2 via `PixelShuffle(2)` outputting $256\times 256$ continuous Radon measures.
  - Alleviates micro-head spatial overlap ($< 4\text{ px}$) on dense crowd scenes.
- **Unit Test Suites (`tests/test_rep_aspp_neck.py`, `tests/test_v50_architecture.py`, `tests/test_v50_components.py`)**:
  - Exact algebraic equivalence test for `switch_to_deploy()`.
  - Shape preservation tests on regular ($512\times 512$) and arbitrary ($409\times 513$) resolutions.
  - Parameter budget assertions and forward/loss/backward verification.
  - Loss orthogonality validation ensuring zero auxiliary interference.
  - Decoupled trust region clamping recovery & non-negativity verification.
- **Granular Experiment Evaluation Artifacts (`rmr_v3/checkpoint.py`, `rmr_v3/engine.py`, `rmr_v3/trainer.py`)**:
  - Automatically exports full per-sample test predictions (`eval_val/predictions.csv`) alongside `summary.json` upon reaching any new best validation checkpoint.
  - Captures exact sample ID, ground truth count, predicted count, absolute error, squared error, and multi-level GAME metrics ($G_0, G_1, G_2, G_3$) for detailed error analysis across density slices.

### Changed
- **Regional Evidence Gating**:
  - Standardized Hurdle gating to `occupancy` mode in `rmr_v3/model/evidence.py`, ensuring $100\%$ mass conservation on regions with $b \ge 1.0$ while attenuating false positives on $b < 1.0$.
  - Eliminated the destructive `product` mode ($\pi_r \cdot b$) which caused double-counting of zero probability and deleted 25% of crowd mass on dense scenes.
- **Regional Scale Pooling**:
  - Enabled `native_scale_pooling: true` in V50, extracting regional features directly from native FPN resolution levels via fractional box prefix sums in $O(1)$ time, eliminating bilinear interpolation artifacts.
- **Loss Suite Purification**:
  - Pruned competing multi-task losses down to 3 orthogonal components: Spatial Point Allocation ($\mathcal{L}_{\text{alloc}}$), Macro Count ($\mathcal{L}_{\text{count}}$), and Regional Evidence ($\mathcal{L}_{\text{region\_nb}}$).
  - Resolved destructive gradient conflict ($\cos \theta = -0.418$) between Flat DM16 and Count loss.
  - Applied faithful Beta-NB variance decoupling (`regional_nb_beta: 0.5`) to eliminate slope compression on dense boxes.

### Fixed
- **Trust Region Zero-Cell Strangulation (`rmr_v3/solver_trust.py`)**:
  - Decoupled mass addition bound (`bound_addition`) from existing carrier mass (`eff_floor = 0.005`). Empty or low-carrier cells with positive regional count deficits can now recover full physical crowd density per step (e.g. 0.35 to 1.0 people) instead of being throttled to +0.00175 per step (a 200x recovery improvement), while maintaining strict non-negativity bounds on mass removal (`bound_removal = eff_kappa * base_bound`).
- **Autograd Gradient Isolation (`rmr_v3/solver.py`)**:
  - Detached `scale_routing_weights` upon solver entry to guarantee zero autograd gradient leak into continuous scale routing weights from unrolled solver steps.
- **Interface Consistency (`rmr_core/heads.py`)**:
  - Forwarded `scale_weights=scale_weights` to `self.activate()` in `ScaleConditionedFineHead.forward()`, ensuring complete polymorphic consistency with `FineMeasureHead`.
- **DTO Mapping Utility (`rmr_core/types.py`)**:
  - Added explicit `.to_dict()` method on `RMRModelOutput` returning a shallow copy dictionary representation.
- **Unit Test Coverage (`tests/test_solver_clamping.py`)**:
  - Created unit tests verifying unconstrained deficit recovery for zero-carrier cells without background leakage.

### Removed
- **Density Curvature Hack**:
  - Bypassed arbitrary quadratic polynomial $y = \text{Softplus}(z)(1 + \alpha \text{ReLU}(z))$ in `rmr_core/heads.py`, restoring strict optimization convexity to the carrier density head.
- **Ad-hoc Solver Flags**:
  - Disabled heuristic solver patches (`deficit_carrier_seeding`, `scale_seeded_carrier`) in default configurations to guarantee monotonic Lipschitz contractive dynamics.
