---
name: anti-regression-protocol
description: Systematic anti-regression methodology for deep learning and inverse problem research. Enforces architectural coherence, prevents ad-hoc assembly of conflicting components, diagnoses delicate equilibrium breakdowns (e.g. hurdle gating, curvature saturation, windowing conflicts), and provides strict preflight verification to guarantee experiments never regress from established benchmark champions.
---

# Anti-Regression & Architectural Coherence Protocol

A specialized research and engineering skill designed to eliminate cyclical regressions, prevent destructive component interference, and enforce deep architectural coherence in competitive machine learning research.

---

## 1. The Core Scientific Premise: Delicate Equilibrium

In complex neural architectures—especially hybrid architectures combining deep neural feature extraction with unrolled variational inverse problem solvers (like RMR)—**components are strongly coupled in a delicate equilibrium**:

$$\text{Final Estimate } y^* = \mathcal{S}_{\text{unrolled}}\left(y_0(\theta_{\text{fine}}), \, b_r(\theta_{\text{reg}}), \, \mathcal{A}, \, \mathcal{W}\right)$$

### The "False Bug" Trap (Delicate Equilibrium Breakdown)
1. **The Phenomenon**: An engineer or researcher inspects a component in isolation and identifies what appears to be a mathematical flaw (e.g., "Hurdle gating erodes 15–25% of dense crowd mass", or "Curvature $\alpha$ is initialized at $-8.0$ which is in the zero-gradient tail of softplus").
2. **The Naive "Fix"**: The researcher modifies that single component to be mathematically "ideal" (e.g., changing hurdle to strict occupancy, or setting $\alpha_0 = 0.0$).
3. **The Catastrophic Regression**: 
   - In reality, the rest of the system (the loss weighting, the solver step size, the carrier prior $m_0$) was **calibrated around that exact behavior**.
   - Removing the 15–25% hurdle dampening causes regional evidence $b_r$ to explode in magnitude, overpowering the carrier field $y_0$ during SIRT unrolling and causing an immediate **$+12.05$ MAE regression** (from $71.51 \to 83.56$).
   - Activating curvature from epoch 1 causes quadratic term $\alpha y^2$ to blow up dense peak mass, resulting in **$+3.81$ MAE regression** ($71.51 \to 75.32$).
4. **The Golden Rule**: 
   > **NEVER modify a coupled component without simultaneously analyzing its conjugate receiver operator.** An apparent "bug" that has produced the best historical benchmark score is often an implicit regularizer or calibration factor.

---

## 2. Component Coherence Matrix

When proposing any architectural change, evaluate its interaction with the other 4 foundational pillars:

| Component | Upstream Dependency | Downstream Impact | Failure Mode if Decoupled |
|:---|:---|:---|:---|
| **Carrier Field ($y_0$)** | Backbone + ASPP Neck | Warm starts SIRT iterations | If $y_0$ is scaled down while $b_r$ is unscaled, solver diverges. |
| **Regional Evidence ($b_r$)** | Region Head (NB rate $\mu$) | Drives SIRT adjoint gradient $\mathcal{A}^* v$ | If $b_r$ has negative bias, solver undercounts. If $b_r$ is uncalibrated, solver overshoots. |
| **Windowing / Boxes ($\mathcal{A}$)** | Grid coordinates / Prefix sums | Defines pooling domain for $A y$ | Fixed square boxes $[32, 64, 128]$ px at tilted horizons contain 100x scale variation, creating conflicting spatial constraints. |
| **Unrolled Solver ($\mathcal{S}$)** | $y_0, b_r, \mathcal{A}$, Morozov deadband | Outputs reconstructed measure $y^*$ | Over-regularization with symmetric deadband causes $-83$ heads negative bias. Disabling deadband lets noise explode. |
| **Loss Function ($\mathcal{L}$)** | Target annotations ($y^{\text{gt}}$) | Direct supervision on $y^*$ and $y_0$ | Multiple primary supervision heads (DM16 + Cell Loss + ChfL) cause gradient thrashing and regression. |

---

## 3. Strict Pre-Flight Anti-Regression Checklist

Before any new experiment configuration is submitted, run, or merged, verify all 6 gates:

- [ ] **Gate 1: True Baseline Identity Check**
  - Is the proposed config derived strictly from the **single highest-performing run** (e.g. `sub60_e5` at $71.51$ MAE)?
  - Are you comparing against the actual checkpoint numbers from `summary.json`, not estimated numbers?
- [ ] **Gate 2: Single-Variable Isolation Invariant**
  - Does the new experiment modify **EXACTLY ONE** logical component relative to the champion baseline?
  - If a compound experiment is proposed, has each constituent component been proven beneficial in isolation first?
- [ ] **Gate 3: Delicate Equilibrium Impact Analysis**
  - Does this change increase or decrease the effective scale of gradients entering the backbone?
  - Does it alter the mass balance between regional evidence $b_r$ and carrier field $y_0$?
  - If changing an activation or initialization (e.g. `alpha_init`, `tau`, `kappa`), did historical runs already test this value? (Always check the historical registry before testing!)
- [ ] **Gate 4: Windowing & Aspect Ratio Invariance**
  - Does the change account for perspective distortion (horizon vs foreground)?
  - Does it introduce boundary artifacts when images are cropped at $512 \times 512$?
- [ ] **Gate 5: Parameter Budget Compliance**
  - Is the model strictly $\le 105,000$ trainable parameters? (Verified via Python runtime parameter count, not guesswork).
- [ ] **Gate 6: Zero Knowledge Distillation & Benchmark Invariants**
  - Is the run strictly $0.0\%$ KD?
  - Are the splits strictly $300$ train (`sha_a_train_all.jsonl`) and $182$ test (`sha_a_test.jsonl`)?

---

## 4. Diagnostic Post-Mortem Protocol When Regression Occurs

When an experiment regresses (e.g. MAE increases from $71.51 \to 75+$):

1. **Decompose the Error by Subgroup**:
   - Check `mae_sparse` ($N \le 100$), `mae_moderate` ($100 < N \le 500$), and `mae_dense` ($N > 500$).
   - Did the regression occur in dense crowds (saturation/explosion) or sparse crowds (false positive background hallucinations)?
2. **Inspect Bias**:
   - Is `Bias` positive (overcounting, background noise) or negative (undercounting, mass erosion, deadband blockage)?
3. **Inspect Solver Trajectory**:
   - Look at `solver_help_fraction` vs `solver_harm_fraction`.
   - In `sub60_e5`, solver helped on **$63.19\%$** of images. If a new config causes `solver_help_fraction` to drop below $55\%$, the solver operator is being fed corrupted evidence.
4. **Revert and Re-isolate**:
   - Do not layer further fixes on top of a regressed model.
   - Revert back to the clean champion baseline (`sub60_e5`), identify the exact offending parameter, and document the failure mode in the repository post-mortem.
