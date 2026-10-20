# Reviewer Memory: Autonomous Research & Review Loop
**Run ID**: `run_20261020_rmr_50mae`
**Topic**: First-Principles Radon Measure Recovery for Sub-50 MAE Crowd Counting on ShanghaiTech Part A
**Current Round**: 2 (Pre-training Verification Complete)
**Reviewer Verdict**: ACCEPT (Score: 8.5 / 10.0)

---

## 1. Resolved Invariants & Mathematical Fixes (Round 1 -> Round 2)
1. **Un-attenuated Beta-NLL Gradient Score ($\beta=1.0$)**:
   - Fixed `rmr_v3/losses/regional.py` to remove unintended batch normalization `w_beta / w_mean`.
   - Elementwise variance-detached weighting restores $\frac{\partial \mathcal{L}_{\beta=1}}{\partial \mu} \equiv \mu - b$, guaranteeing an exact linear slope across dense crops ($N > 1000$).
2. **Support Absorption & Upward Transport Headroom**:
   - Configured `rn_trust_alpha_up: 1.5 - 2.0` in `configs/rmr_v60/`, allowing the closed-form mass transport operator to deliver full mass to dense regions without getting choked by symmetric 15% bounds.
3. **Loss Purification & Minimal 3-Loss Suite**:
   - Removed all 6 conflicting auxiliary losses ($\lambda_{\text{cell}}=0, \lambda_{\text{hurdle}}=0, \lambda_{\text{trunc}}=0, \lambda_{\text{scale\_align}}=0, \lambda_{\text{curv}}=0, \lambda_{\text{hard\_bg}}=0$).
   - Retained only 3 mutually orthogonal objectives: Count Poisson/NB ($\lambda=1.0$), Allocation DM16/ChfL ($\lambda=1.0 / 0.35$), and Regional Beta-NB ($\lambda=0.20$).
4. **Subpixel Stride 2 via Zero-Parameter PixelShuffle**:
   - Track B configuration slashes cell collision from 38.3% to 4.1% at 144,960 params ($\le 200,000$).
5. **Strict Training Protocol**:
   - Strictly 1,000 epochs with Cosine Annealing.
   - Permanent ban on WSD and 650-epoch premature decay.
   - 100% Standalone (0.0% Knowledge Distillation).
6. **Code Quality & Test Integrity**:
   - All files $\le 450$ lines.
   - 142/142 tests passing across mathematical essence, adversarial stress, and unit suites.

---

## 2. Active Configs Ready for Execution
- `configs/rmr_v60/v60_track_a_closed_form_rn.yaml` (103,959 params)
- `configs/rmr_v60/v60_track_b_subpixel_stride2_rn.yaml` (144,960 params)
- `configs/rmr_v60/v60_track_b_chfl_subpixel_rn.yaml` (144,960 params)
