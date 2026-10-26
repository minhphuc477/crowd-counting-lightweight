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
5. **Zero Frankencode & Zero Meaningless Patchwork**:
   - Strictly forbidden to arbitrarily cobble together heuristics, dilated layers, or meaningless patches. Everything must have rigorous mathematical/physical justification.
6. **Controlled Experimentation Protocol**:
   - Never create random experiments on a whim ("không phải muốn là tạo"). Every proposed experiment must have a clear single-variable control against the champion baseline (V64, 69.03 MAE) and require explicit user consent.

## 4. Key Artifacts & Narrative Reports
- Anti-Regression Experimental Guidelines: `LUU_Y_THIET_KE_THI_NGHIEM_ANTI_REGRESSION.md`
- Operational Rules: `RULES.md`
- Latest V65 Full Ubuntu Analysis: `BAO_CAO_KHOA_HOC_PHAN_TICH_TOAN_DIEN_KET_QUA_V65_UBUNTU.md`
- Latest V65 Regression Root Cause & Cleanup: `BAO_CAO_GIAI_PHAU_NGUYEN_NHAN_REGRESSION_V65_VA_CLEANUP_CODEBASE.md`
- Active Research Branch: `RMR`
- Active Configs: `configs/rmr_v64/`, `configs/rmr_v65/`
