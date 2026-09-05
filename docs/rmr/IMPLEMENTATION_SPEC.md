# RMR-Count: Implementation Specification & Engineering Guide

> **Module:** `rmr_count`  
> **Repository:** `lightweightcrcn` (branch: `RMR`)  
> **Environment:** Python 3.10+, PyTorch 2.4+, CUDA 12.x / CPU compatible.

---

## 1. Codebase Structure

```
rmr_count/
├── __init__.py           # Package exports
├── model.py              # TinyLocalEncoder, AdditiveFusion, Fine/Regional Heads, Unrolled Reconciler
├── operators.py          # prefix2d, regional_sum, regional_adjoint, RegionSet geometry caching
├── losses.py             # LossConfig, compute_losses (balanced cell & regional rate Huber losses)
├── data.py               # CrowdManifestDataset, Low-RAM PIL cropping, rasterize_points
├── eval.py               # Evaluator, Tiled & Direct Inference, Diagnostic Traces
├── metrics.py            # Canonical NAE, Physical-support GAME(0..3), MAE, RMSE
├── profile.py            # Config-driven FP32 & AMP Latency, Peak VRAM, FLOPs Profiler
├── aggregate.py          # Multi-seed mean/std, Paired Bootstrap 95% CI comparisons
└── prepare_manifest.py   # Dataset preprocessor: SHA, UCF-QNRF, NWPU multi-format

# Root Scripts
run_rmr_matrix.sh         # Matrix execution script for registered benchmark B0–B5 across 3 seeds
run_lr_sweep.ps1          # Pilot learning rate sweep script (1e-4, 3e-4, 1e-3)
```

---

## 2. Model Architecture & Exact Parameter Counts

The concrete model architecture consists of:
- **`TinyLocalEncoder` (~52k params):** Native local-first convolutional backbone.
  - Stem: `ConvGNAct(3 -> 16, k=3, stride=2)`
  - $C_4$ stage: `TinyIR(16 -> 24, stride=2)` + `TinyIR(24 -> 24)`
  - $C_8$ stage: `TinyIR(24 -> 40, stride=2)` + 2x `TinyIR(40 -> 40)`
  - $C_{16}$ stage: `TinyIR(40 -> 64, stride=2)` + `TinyIR(64 -> 64)`
- **`AdditiveFusion` Neck (~3.5k params, width=32):**
  - $1 \times 1$ projections of $C_4, C_8, C_{16}$ to 32 channels.
  - Bilinear interpolation to $C_4$ spatial resolution followed by additive fusion.
  - Depthwise-separable $3 \times 3$ `ConvGNAct` + $1 \times 1$ `ConvGNAct`.
- **`FineMeasureHead` (~3.2k params, width=32):**
  - Depthwise-separable conv + Conv $1 \times 1$.
  - Calibrated bias init $\approx -4.595$ yielding initial count rate $\operatorname{softplus}(-4.595) \approx 0.01$ count/cell.
- **`RegionalMeasureHead` (~4.2k params):**
  - Multi-scale ROI-pooling on bounding boxes $\{32, 64, 128\}$ px with 4D geometry $[ \log h, \log w, \log |R|, \log(w/h) ]$.
  - Predicts regional rate $\rho_R$ such that $b_R = |R| \cdot \rho_R$.
- **Unrolled Reconciliation Layer (~1.5k params):**
  - Parameterized step size $\eta_t = \eta_{\text{max}} \cdot \sigma(\alpha_t)$ with learnable logits $\alpha_t$, initialized to $\eta_t(0) = \eta_{\text{init}} = 0.05$ (with $\eta_{\text{max}} = 0.20$).
  - Preconditioner block $M^{(t)}$ producing local confidence weights in $[0, 1]$.

### Verified Parameter Counts:
Exact values returned by `count_parameters(model)`:
- **B0 (`direct`):** 58,867
- **B1 (`region_loss`):** 58,867
- **B2 (`region_aux`):** 63,044 (+4,177 params from regional head)
- **B3a (`local_refine`):** 61,876 (+3,009 params from local recurrent refinement)
- **B3b (`learned_project`):** 66,086 (+3,042 params from neural membership projector)
- **B4 (`rmr_t1`, $T=1$):** 64,580 (+1,536 preconditioner + 1 step logit)
- **B5 (`rmr_t2`, $T=2$):** 64,581 (+1,536 preconditioner + 2 step logits)

---

## 3. Mathematical Operators & Memory Safeguards

### 3.1 2D Prefix Sum (`prefix2d`)
Fast $O(1)$ rectangle count querying uses 2D integral images with AMP float32 safety:
```python
def prefix2d(x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    if x.is_cuda and torch.is_autocast_enabled():
        x = x.float()
    p = torch.cumsum(torch.cumsum(x, dim=-1), dim=-2)
    p = F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)
    return p.to(dtype=orig_dtype)
```

### 3.2 Geometry & Coverage Caching
1. `model._regions_and_coverage(h, w)` caches $(RegionSet, D_c)$ where $D_c = A^\top \mathbf{1}_M$. In iterative solver loops ($T=2$), this eliminates redundant adjoint calls, saving 8 `index_add_` and 4 cumsum operations per forward pass.
2. `RegionSet.boxes_list` is pre-cached as Python tuples, preventing GPU $\leftrightarrow$ CPU device synchronization during B3b (`LearnedMembershipProjector`) execution.

### 3.3 Low-RAM PIL Data Pipeline
To prevent Windows memory thrashing and `ArrayMemoryError`:
1. Images are loaded as PIL RGB images (~768 KB uint8).
2. Random scale resizing (0.75x–1.25x) and 512x512 random cropping execute strictly within PIL uint8 space.
3. Only the final 512x512 crop is converted to a PyTorch tensor and normalized with ImageNet statistics (~3 MB peak per sample).

---

## 4. Loss Formulation & Variant Dispatch

Every variant is trained under matched loss objectives via `compute_losses(outputs, target_y, variant, cfg)`:

### 4.1 Fine Cell & Global Losses
$$\mathcal{L}_{\text{cell}} = \operatorname{SmoothL1}(Y, Y^*; \; \beta = 1.0), \qquad \mathcal{L}_{\text{global}} = |N_{\text{pred}} - N_{\text{gt}}|.$$

### 4.2 Regional Rate Loss (Scale-Balanced Huber)
$$\mathcal{L}_{\text{region}}(b, N^*) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \frac{1}{|R_s|} \sum_{m \in R_s} \operatorname{SmoothL1}\left(\frac{b_m}{|R_m|}, \; \frac{N_m^*}{|R_m|}; \; \beta = 0.1\right).$$

### 4.3 Loss Dispatch Matrix
- **B0 (`direct`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}}$
- **B1 (`region_loss`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}} + 0.2 \cdot \mathcal{L}_{\text{region}}(A Y, N^*)$
- **B2 (`region_aux`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}} + 0.2 \cdot \mathcal{L}_{\text{region}}(b, N^*)$
- **B3a (`local_refine`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}}$
- **B3b (`learned_project`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}} + 0.2 \cdot \mathcal{L}_{\text{region}}(b, N^*)$
- **B4 / B5 (`rmr_t1` / `rmr_t2`):** $\mathcal{L} = \mathcal{L}_{\text{cell}} + 0.1 \cdot \mathcal{L}_{\text{global}} + 0.2 \cdot \mathcal{L}_{\text{region}}(b, N^*)$

*Critical Scientific Property:* RMR does **not** receive $\mathcal{L}_{\text{region\_map}}$ on the output map $AY$. Fine map reconciliation occurs entirely via the unrolled forward operator dynamics, ensuring a clean causal comparison with B2 and B3b. Deep supervision is disabled by default ($\lambda_{\text{deep\_supervision}} = 0.0$).

---

## 5. Training Protocol & Schedules

- **Registered Benchmark Matrix:**
  - Configs: `configs/rmr/*.yaml`
  - Total Epochs: **1000**
  - Learning Rate: CosineAnnealingLR with 5-epoch linear warmup.
  - Evaluation Schedule: Validation every **10 epochs**.
  - Checkpoint Rule: `best_val_mae.pt` updated only when `solver_strength == 1.0`.
  - Gradient Clipping: $\text{clip\_norm} = 5.0$.
  - Precision: Automatic Mixed Precision (AMP).

- **Pilot Learning Rate Sweep:**
  - Script: `run_lr_sweep.ps1`
  - Total Epochs: **100**
  - Evaluation Schedule: Validation every **5 epochs**.
  - Grid: $\eta \in \{1\text{e-}4, 3\text{e-}4, 1\text{e-}3\}$.
