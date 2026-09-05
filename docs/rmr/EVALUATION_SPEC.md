# RMR-Count: Evaluation Specification & Diagnostic Rigor

> **Method:** Regional Measure Reconciliation (RMR-Count)  
> **Evaluation Standards:** Adheres to CVPR / ECCV / PAMI canonical evaluation protocols for visual crowd counting.  
> **Evaluation Scope:** RMR-Count is evaluated strictly as a visual crowd counting and spatial density estimation model. Grid Average Mean Error (GAME) measures multi-scale spatial count discrepancy across rigid spatial partitions; it is not a point localization or detection metric. The system does not output point coordinates or compute localization Precision / Recall / F1 scores.

---

## 1. Canonical Evaluation Metrics

### 1.1 Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{N}_i - N_i|, \qquad \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{N}_i - N_i)^2}.$$

### 1.2 Normalized Absolute Error (NAE)
To prevent division-by-zero distortion on empty background scenes, NAE is strictly evaluated over the non-empty image subset $\mathcal{I}_+ = \{i : N_i > 0\}$:
$$\text{NAE} = \frac{1}{|\mathcal{I}_+|} \sum_{i \in \mathcal{I}_+} \frac{|\hat{N}_i - N_i|}{N_i}.$$
*(Note: If all images in a test set are zero-GT, NAE returns NaN).*

### 1.3 Physical-Support Grid Average Mean Error (GAME)
Conventional grid partitions on output feature maps suffer from edge quantization when $H_o, W_o$ are not divisible by $2^L$.  
RMR implements **physical-support GAME**:
1. The original image support $[0, W] \times [0, H]$ is partitioned into $4^L$ exact rectangles ($2^L \times 2^L$).
2. Model prediction cells $[c_{x0}, c_{x1}] \times [c_{y0}, c_{y1}]$ deposit fractional mass into partitions based on geometric 2D bounding box overlap weights $W_Y \hat{Y} W_X^\top$.
3. Ground-truth point annotations $\mathcal{P}_i$ are binned directly into the physical partitions.
4. **Mathematical Guarantee:** At level $L=0$, $\text{GAME}(0) \equiv |\hat{N} - N| \equiv \text{MAE}$ exactly, with zero edge quantization loss.

---

## 2. Mechanism Tracing & Diagnostic Artifacts

Every standalone evaluation run (`python -m rmr_count.eval --checkpoint <path> --manifest <path> --out-dir <dir>`) automatically writes structured long-table diagnostic artifacts:

### 2.1 `solver_trace.csv` (Per-Iteration Reconciliation Dynamics)
Records the trajectory of the unrolled reconciliation layer at each step $t \in \{0, \dots, T-1\}$:
- `image_id`: Unique sample identifier.
- `iteration`: Step index $t$.
- `eta`: Effective step size $\eta_t$.
- `energy_before`: $\mathcal{E}_a(Y^{(t)}) = \frac{1}{2} \sum_{m=1}^M \frac{(q_m^{(t)} - b_m)^2}{|R_m|}$.
- `energy_after`: $\mathcal{E}_a(Y^{(t+1)})$.
- `residual_mean`: Average magnitude of the rate residual field $|\nabla_Y \mathcal{E}_a|$.
- `residual_max`: Peak rate residual magnitude.
- `clip_fraction`: Percentage of cells whose updates hit the clipping threshold $\tau = 5.0$.
- `preconditioner_mean`: Average gating magnitude of $M^{(t)}$.
- `preconditioner_max`: Peak gating magnitude.
- `delta_n`: Absolute change in total count across the step $|\sum Y^{(t+1)} - \sum Y^{(t)}|$.
- `delta_l1`: Total absolute cell displacement $\|Y^{(t+1)} - Y^{(t)}\|_1$.

### 2.2 `regional_trace.csv` (Multi-Scale Region Residuals)
Records region-level predictions against ground truth for mechanistic analysis:
- `image_id`: Sample identifier.
- `region_idx`: Index of the bounding box.
- `scale`: Window size in pixels ($32, 64, 128$).
- `area`: Box area in output cells $|R_m|$.
- `gt_count`: Ground-truth points inside $R_m$.
- `b_pred`: Regional head prediction $b_m$.
- `q_pred`: Fine density integral $q_m = (A Y)_m$.
- `count_residual`: $q_m - b_m$.
- `rate_residual`: $\frac{q_m - b_m}{|R_m|}$.

### 2.3 `predictions.csv` & `summary.json`
- `predictions.csv`: Per-image predictions, errors, and GAME0–3 metrics for direct and tiled inference.
- `summary.json`: Macro metrics, density-stratified MAE, Pearson correlation between $b_R$ and ground-truth regional counts, and solver energy reduction rates.

---

## 3. Statistical Significance & Paired Comparisons

To verify whether RMR (B5) statistically outperforms baselines rather than benefitting from stochastic seed variation, `rmr_count/aggregate.py` performs paired image-level analyses:

$$d_i = |\hat{N}_i^{\text{treatment}} - N_i| - |\hat{N}_i^{\text{control}} - N_i|.$$

The tool computes:
1. **Mean Difference $\bar{d}$:** Negative values indicate treatment reduces absolute count error.
2. **Bootstrap 95% Confidence Interval:** Empirical percentile bootstrap over 10,000 resamples: $[\text{CI}_{\text{lo}}, \text{CI}_{\text{hi}}]$.
3. **Paired Two-Tailed t-test:** Testing null hypothesis $H_0: \mathbb{E}[d] = 0$.
4. **Wilcoxon Signed-Rank Test:** Non-parametric test robust to heavy-tailed crowd count outliers.
5. **Win / Loss / Tie Counts:** Sample-level tally across the test set.

CLI Usage:
```powershell
python -m rmr_count.aggregate --compare runs/sha_a/rmr_t2_seed42/predictions.csv runs/sha_a/direct_seed42/predictions.csv --name-a B5_RMR --name-b B0_Direct
```

---

## 4. Latency & Resource Profiling Protocol

`rmr_count/profile.py` provides standardized timing and complexity metrics:
- **Clean Single-Forward Peak VRAM:** Measured via `reset_peak_memory_stats() -> synchronize() -> model(x) -> synchronize() -> max_memory_allocated()` prior to loop execution.
- **Latency & FPS:** Measured across 50 warmup iterations and 200 timed iterations using CUDA events for both FP32 and AMP (`torch.amp.autocast('cuda')`).
- **Complexity Notes:** Outputs `profiler_supported_flops` with explicit documentation that PyTorch profiler measures convolution and linear operations, while prefix-sum cumsum and scatter-add operations are $O(G + M)$ memory-bound primitives.
