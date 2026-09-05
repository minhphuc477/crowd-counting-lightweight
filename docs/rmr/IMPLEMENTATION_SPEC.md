# RMR-Count: Implementation Specification & Engineering Guide

> **Module:** `rmr_count`  
> **Repository:** `lightweightcrcn` (branch: `RMR`)  
> **Environment:** Python 3.11+, PyTorch 2.6+, CUDA 12.x / CPU compatible.

---

## 1. Codebase Architecture

```
rmr_count/
├── __init__.py           # Package exports
├── model.py              # RMRCount, RMRConfig, Heads, Solvers, Preconditioners
├── operators.py          # prefix2d, regional_sum, regional_adjoint, RegionSet
├── losses.py             # LossConfig, compute_losses, cell & rate Huber losses
├── data.py               # CrowdManifestDataset, Low-RAM PIL cropping, transforms
├── eval.py               # Evaluator, Tiled & Direct Inference, Mechanism Traces
├── metrics.py            # Canonical NAE, Physical-support GAME(0..3), MAE, RMSE
├── profile.py            # FP32 & AMP Latency, Peak VRAM, FLOPs Profiler
├── aggregate.py          # Multi-seed mean/std, Paired Bootstrap 95% CI comparisons
├── prepare_manifest.py   # Dataset preprocessor: SHA, UCF-QNRF, NWPU multi-format
└── runner.py             # Registered experiment matrix coordinator (B0–B5)
```

---

## 2. Mathematical Operators & Memory Optimizations

### 2.1 2D Prefix Sum (`prefix2d`)
Fast $O(1)$ rectangle count querying uses discrete 2D integral images:
```python
def prefix2d(x: torch.Tensor) -> torch.Tensor:
    # Autocast FP32 accumulation guard: prevent float16 overflow in sum
    orig_dtype = x.dtype
    if x.is_cuda and torch.is_autocast_enabled():
        x = x.float()
    p = torch.cumsum(torch.cumsum(x, dim=-1), dim=-2)
    # Zero-padding top and left for 1-based indexing
    p = F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)
    return p.to(dtype=orig_dtype)
```

### 2.2 Regional Sum & Adjoint Scatter
- `regional_sum(p, boxes)` evaluates $(A Y)_m$ for all $M$ boxes in $O(M)$ time using the 4 corners of $p$.
- `regional_adjoint(r, boxes, h, w)` scatters regional values onto the spatial grid using `index_add_`.
- **AMP FP32 Guard:** In `regional_adjoint`, accumulator buffers are forced to `torch.float32` under CUDA autocast to avoid non-associative half-precision accumulation errors.

### 2.3 Geometry & Coverage Caching
Because the regional bounding boxes $\mathcal{R}$ and cell coverage $D_c = A^\top \mathbf{1}_M$ depend only on the spatial grid dimensions $(H_o, W_o)$ and scale definitions:
1. `model._regions_and_coverage(h, w)` caches $(RegionSet, D_c)$ across iterations.
2. In iterative loops ($T=2$), this eliminates duplicate `regional_adjoint` calls for coverage computation, saving 8 `index_add_` and 4 cumsum operations per forward pass.
3. `RegionSet.boxes_list` is pre-cached as Python tuples to prevent GPU $\leftrightarrow$ CPU device synchronization during B3b (`LearnedMembershipProjector`) execution.

---

## 3. Data Pipeline & Low-RAM Footprint

To ensure stable execution on memory-constrained systems (e.g., 16 GB RAM with low commit limits on Windows), `rmr_count/data.py` executes all spatial augmentations in **PIL uint8 space**:

```python
# Low-RAM Pipeline (Peak: ~1.5 MB per image)
1. PIL.Image.open(img_path).convert("RGB")  # ~768 KB (uint8)
2. Random scale resize (0.75x - 1.25x) in PIL.Image.Resampling.BILINEAR
3. Random 512x512 crop directly on PIL Image
4. Scale and shift annotation points: p_new = (p * scale) - crop_offset
5. Convert ONLY the 512x512 crop to torch.Tensor and normalize with ImageNet stats
```

**Savings:** Eliminates converting full-resolution images ($1000 \times 750 \times 3 \times 4$ bytes $\approx 9$ MB) to float32 tensors before cropping, preventing Python CLR memory thrashing and Windows `ArrayMemoryError`.

---

## 4. Loss Formulation & Calibration

### 4.1 Fine Cell Loss
Local prediction $Y$ is supervised by ground-truth cell counts $Y^*$ using Smooth L1 (Huber) loss:
$$\mathcal{L}_{\text{cell}} = \operatorname{SmoothL1}(Y, Y^*; \; \beta = 1.0).$$

### 4.2 Regional Rate Loss
Supervising counts directly would cause gradients to scale proportionally with region area ($|R| \in [64, 1024]$), destabilizing multi-scale training. We supervise **density rates**:
$$\rho_m = \frac{b_m}{|R_m|}, \qquad \rho_m^* = \frac{N_m^*}{|R_m|}.$$
$$\mathcal{L}_{\text{region}} = \frac{1}{M} \sum_{m=1}^M \operatorname{SmoothL1}(\rho_m, \rho_m^*; \; \beta = 0.1).$$

Using $\beta = 0.1$ maintains non-trivial residuals in the linear regime, preventing gradient decay on sparse regions.

### 4.3 Total Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cell}}(Y_T, Y^*) + \lambda_{\text{global}} |N_T - N^*| + \lambda_{\text{region\_head}} \mathcal{L}_{\text{region}}(b, N_{\mathcal{R}}^*) + \lambda_{\text{region\_map}} \mathcal{L}_{\text{region}}(A Y_T, N_{\mathcal{R}}^*).$$
Default hyperparameters:
$$\lambda_{\text{global}} = 0.10, \quad \lambda_{\text{region\_head}} = 0.20, \quad \lambda_{\text{region\_map}} = 0.20, \quad \lambda_{\text{deep\_supervision}} = 0.0.$$

---

## 5. Model Configuration Reference

A canonical RMR model config (`configs/rmr/rmr_t2.yaml`):

```yaml
seed: 42
model:
  variant: rmr
  output_stride: 4
  feature_width: 32
  region_sizes_px: [32, 64, 128]
  region_overlap: 0.5
  include_full_image: false
  iterations: 2
  eta_init: 0.05
  eta_max: 0.20
  residual_clip: 5.0
  use_jacobian_gate: false

train:
  lr: 3.0e-4
  weight_decay: 1.0e-4
  epochs: 60
  warmup_epochs: 5
  solver_warmup_epochs: 5
  grad_clip: 5.0
  amp: true
  workers: 0
  pin_memory: false
```
