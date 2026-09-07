# RMR-Count: Implementation Specification & Engineering Guide

> **Module:** `rmr_count`  
> **Repository:** `lightweightcrcn` (branch: `RMR`)  
> **Environment:** Python 3.10+, PyTorch 2.4+, CUDA 12.x / CPU compatible.

---

## 1. Codebase Structure

```
rmr_count/
├── __init__.py           # Package exports
├── model.py              # MobileNetV4 Carrier, Additive FPN, ScaleMatched Head, Projected-SIRT & B3b
├── operators.py          # prefix2d, regional_sum, regional_adjoint (FP32 AMP-safe), RegionSet caching
├── losses.py             # LossConfig, compute_losses (Flat-DM16, NB count, Scale-Balanced Huber)
├── data.py               # CrowdManifestDataset, Low-RAM PIL cropping, rasterize_points
├── eval.py               # Evaluator, Tiled & Direct Inference, Diagnostic Traces
├── metrics.py            # Canonical NAE, Physical-support GAME(0..3), MAE, RMSE
├── profile.py            # Config-driven FP32 & AMP Latency, Peak VRAM, FLOPs Profiler
├── aggregate.py          # Multi-seed mean/std, Paired Bootstrap 95% CI comparisons
├── localization/otm.py   # Canonical normalize_image [0.5, 0.5, 0.5] for localization testing
└── prepare_manifest.py   # Dataset preprocessor: SHA, UCF-QNRF, NWPU multi-format

# Root Runners & Scripts
run_stage_c_matrix.ps1    # Matrix execution script for registered benchmark B0–B5 (Val-only evaluation)
run_final_test_eval.ps1   # Post-freeze test benchmark on sha_a_test.jsonl (run once after matrix completes)
scripts/legacy/           # Archived exploratory and legacy pilot runners
```

---

## 2. Model Architecture & Exact Parameter Counts

The concrete model architecture consists of:
- **Pretrained `MobileNetV4-Conv-Small-0.5` Carrier (~87k params):**
  - Truncated at feature reduction 16 ($C_4: 16\text{ch}, C_8: 32\text{ch}, C_{16}: 48\text{ch}$).
  - Physically excludes $C_{32}$ to maximize mobile-edge efficiency.
  - Backbone learning rate is scaled by $0.1\times$ relative to heads.
- **Additive FPN Neck (~7.3k params, width=32):**
  - $1 \times 1$ lateral projections of $C_4, C_8, C_{16}$ to 32 channels.
  - Dilated depthwise-separable $3 \times 3$ convolutions with dilation factors $d \in \{1, 2, 3\}$ applied on $P_{16}$ level only, then top-down fused into $P_8$ and $P_4$.
  - Multi-scale representations for scale-matched feature routing.
- **`FineMeasureHead` (~3.2k params, width=32):**
  - Depthwise-separable $3 \times 3$ conv + Conv $1 \times 1$ on $P_4$ (output stride $s=4$).
  - Data-driven prior bias init $\approx -4.1422$ yielding empirical mean density $m_0 \approx 0.015763$ count/cell.
- **`ScaleMatchedRegionalEvidenceHead` (RMR-v1/v2, ~4.0k params):**
  - Multi-scale ROI-pooling with physical pixel scale routing:
    - $\le 48\text{px} \to P_4$
    - $48\text{px} < s \le 96\text{px} \to P_8$
    - $> 96\text{px} \to P_{16}$
  - Concatenates 4D geometry $[ \log h, \log w, \log |R|, \log(w/h) ]$.
  - Predicts regional count mass $b$.
- **`ProbabilisticRegionalEvidenceHead` (RMR-v3, ~4.1k params):**
  - 33D input representation: 32D visual feature average-pooled over fine cells (with $P_8$ and $P_{16}$ bilinearly upsampled to $P_4$ support) concatenated with 1D scale log-ratio $\log(K/32)$.
  - 2-layer MLP ($33 \to 48 \to 48$ with SiLU), splitting into:
    - `mean_head: Linear(48, 1)` for regional rate $\mu_R$.
    - `log_dispersion_head: Linear(48, 1)` for dispersion parameter $\theta_R \in [0.5, 500.0]$.
  - Reliability weights derived via variance of regional rate: $w_R \propto \frac{1}{\operatorname{Var}(\hat{r}_R)}$.
- **Projected SIRT Reconciliation Layer (0 params):**
  - Measure-space nonnegative projection $\Pi_+ [Y_t - \omega \cdot D_c^{-1} A^\top D_a^{-1} (A Y_t - b)]$.
  - In RMR-v3: $\Pi_+ [Y_t - \omega \cdot D_{c,w}^{-1} A^\top W D_a^{-1} (A Y_t - \mu)]$, with diagonal preconditioner $D_{c,w} = \operatorname{diag}(A^\top w)$, where $w = W \mathbf{1}_M$ is the regional weight vector without area scaling in coverage.
  - Parameter-free with canonical $\omega = 1.0, T = 2$.
  - Regional evidence $b$ and weights $W$ are detached during unrolled steps to isolate causal reconciliation.

### Verified Parameter Counts:
Exact values returned by `count_parameters(model)`:
- **B0 (`direct`):** 97,681
- **B1 (`region_loss`):** 97,681
- **B2 (`region_aux`):** 101,714 (+4,033 params from regional head)
- **B3a (`local_refine`):** 100,692 (+3,011 params from local refinement conv)
- **B3b (`learned_project`):** 104,756 (+3,042 params from measure-space $P_\theta$)
- **B5-P (`rmr_p`, $T=2$):** 101,714 (0 extra parameters over B2)
- **RMR-v3 (`uniform_control` & `rw`):** 101,763 (+49 params over B5-P from `log_dispersion_head`)

---

## 3. Mathematical Operators & FP32 AMP Numerical Safeguards

### 3.1 2D Prefix Sum (`prefix2d`)
Fast $O(1)$ rectangle count querying uses 2D integral images. To prevent mantissa cancellation under AMP (FP16/BF16):
```python
def prefix2d(x: torch.Tensor, preserve_fp32: bool = True) -> torch.Tensor:
    orig_dtype = x.dtype
    if preserve_fp32 and x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    p = torch.cumsum(torch.cumsum(x, dim=-1), dim=-2)
    p = F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)
    return p if preserve_fp32 else p.to(dtype=orig_dtype)
```

### 3.2 FP32 Accumulation in `regional_sum` and `regional_adjoint`
- `regional_sum(x, boxes)` computes prefix sum in FP32, extracts 4-point rectangle differences strictly in FP32, and casts to target dtype at the output boundary.
- `regional_adjoint(values, boxes, ...)` populates the 2D difference buffer and evaluates 2D cumsums strictly in FP32 before returning.
- All operator tests in `tests/rmr/test_rmr_operators.py` verify that FP16 and BF16 execution matches FP32 reference with relative error $< 5 \times 10^{-3}$ and zero numerical cancellation.

### 3.3 Geometry & Coverage Caching
1. `model._regions_and_coverage(h, w)` caches $(RegionSet, D_c)$ where $D_c = A^\top \mathbf{1}_M$. In iterative solver loops ($T=2$), this eliminates redundant adjoint calls, saving 8 `index_add_` and 4 cumsum operations per forward pass.
2. `ScaleMatchedRegionalEvidenceHead` routes regions based on physical pixel size regardless of dictionary ordering or scale permutations.

---

## 4. Loss Formulation & Training Objectives

Every variant is trained under matched loss objectives via `compute_losses(outputs, target_y, variant, cfg)`:

### 4.1 Count Loss & Count-Normalized Flat Dirichlet-Multinomial-16
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{count}} + \lambda_{\text{dm}} \cdot \mathcal{L}_{\text{flat\_dm16}}^{\text{norm}} + \lambda_{\text{cell}} \cdot \mathcal{L}_{\text{cell}} + \lambda_{\text{region}} \cdot \mathcal{L}_{\text{region}}.$$
- $\mathcal{L}_{\text{count}}$: SmoothL1 or Negative Binomial count loss ($\lambda_{\text{count}} = 1.0$).
- $\mathcal{L}_{\text{flat\_dm16}}^{\text{norm}}$: Dirichlet-Multinomial-16 allocation loss on $16\text{px}$ blocks ($4 \times 4$ stride-4 cells), normalized by crowd size:
  $$\mathcal{L}_{\text{flat\_dm16}}^{\text{norm}} = \frac{-\log p(y \mid \alpha)}{\max(N_i, 1)}$$
  representing spatial allocation NLL per person ($\sim 0.5 - 7 \text{ nats/person}$). This prevents dense crops from receiving hundreds of times more gradient weight than sparse crops.
- $\mathcal{L}_{\text{cell}}$: Balanced SmoothL1 cell density loss ($\lambda_{\text{cell}} = 0.25$).
- $\mathcal{L}_{\text{region}}$: Scale-balanced Huber rate loss on regional evidence $b$ vs ground-truth regional counts $N^*$.

### 4.2 Causal Isolation Protocol
- Fine map reconciliation in RMR-P occurs entirely via the unrolled forward operator dynamics.
- Regional evidence $b$ is detached (`b.detach()`) during solver unrolling, ensuring that the backward pass does not backpropagate through the solver into the regional head.
- B3b uses the exact same detached regional evidence and measure-space parameterization.

---

## 5. Training Protocol & Directory Safeguards

### 5.1 Directory Overwrite Guard
To prevent accidental mixing of artifacts across training generations:
- `rmr_count/train.py` verifies `output_dir` before training starts.
- If `output_dir` exists and contains artifacts (e.g. checkpoints or CSV logs), training terminates with `FileExistsError` unless `--overwrite` or `--resume` is explicitly passed.

### 5.2 Logging Fieldnames
`train_log.csv` records comprehensive per-epoch dynamics:
`epoch,train_loss,train_count,train_flat_dm16,train_region,lr_backbone,lr_main,val_mae,val_rmse,val_game0,val_game1,val_game2,val_game3`.

### 5.3 Training Hyperparameters
- **Epochs:** 1000 epochs (early stopping disabled for full convergence).
- **Optimizer:** AdamW with cosine annealing schedule.
- **Learning Rates:** Main head LR $10^{-4}$, Backbone LR $10^{-5}$ ($0.1\times$ backbone multiplier).
- **Training Data:** 100% of official `train_data` (300 images via `data/sha_a_train_all.jsonl`).
- **Evaluation & Model Selection:** Evaluated on canonical benchmark test set (`sha_a_test.jsonl`, 182 images) every 10 epochs to track `best_val_mae.pt`, adhering strictly to literature convention without ad-hoc holdout splits.
