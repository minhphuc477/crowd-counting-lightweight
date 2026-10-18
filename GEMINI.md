# GEMINI.md - RMR Autonomous Research & Engineering Configuration

## 1. Remote GPU Server (for autonomous experiments)
- Server OS: Ubuntu Linux (remote workstation)
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB GDDR7 VRAM, SM 12.0)
- Python / PyTorch: PyTorch 2.6+, CUDA 12.x, AMP FP16 supported
- Repo branch: `RMR` (git remote: `origin/RMR`)
- Workspace directory: `/home/user/lightweightcrcn` (or equivalent workstation path)
- Execution pattern: Run directly with `python train.py --config <config_path>` or in background screen/tmux sessions

## 2. Research Project Context
- **Topic**: Lightweight Radon Measure Recovery (RMR) for Crowd Counting on ShanghaiTech Part A
- **Target venue**: CVPR / ICCV / ECCV / NeurIPS / TPAMI (Rank A* / Q1)
- **Mathematical formulation**:
  $$\text{Fredholm Integral Operator: } b = \mathcal{A}_\pi y + \eta$$
  $$\text{Unrolled Adjoint Reconstruction: } y_{t+1} = \mathcal{P}_+\left(y_t + \omega_t \frac{y_t}{\text{cov}_\pi} \mathcal{A}_\pi^T \mathcal{S}_{\gamma \sigma}(b - \mathcal{A}_\pi y_t)\right)$$

## 3. Strict Non-Negotiable Invariants
1. **Parameter Budgets**:
   - Track A (Stride 4): Strictly $\le 104,441$ trainable parameters.
   - Track B (Subpixel Stride 2): Strictly $\le 200,000$ trainable parameters.
2. **100% Standalone**:
   - Exactly $0.0\%$ Knowledge Distillation (KD). Zero teacher dependencies at inference or training.
3. **Empirical Integrity**:
   - Zero fabricated metrics, zero oracle substitutions in reported benchmarks. All test evaluations run on full 182 test images via standard MAE / RMSE / GAME / NAE protocols.
4. **Code Quality**:
   - Strictly $\le 450$ lines per file.
   - Pure Python / YAML configurations.

## 4. Key Artifacts & Narrative Reports
- Current Analysis Artifact: `BAO_CAO_KHOA_HOC_PHAN_TICH_TOAN_DIEN_KET_QUA_V50_UBUNTU.md`
- Active Research Branch: `RMR`
- Configs: `configs/rmr_v50/`, `configs/rmr_v51/`
