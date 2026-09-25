from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.operators import build_multiscale_regions, regional_sum
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_core.evaluation import evaluate_dataset


# =========================================================================
# Test 1: Dual-Lattice Cell Loss Exact Single-Counting
# =========================================================================

def test_dual_lattice_cell_loss_single_counting():
    """Verify that in dual-lattice mode, fine cell loss is never double-counted."""
    h_fine, w_fine = 64, 64
    h_carrier, w_carrier = 32, 32

    y_fine = torch.rand(1, 1, h_fine, w_fine, requires_grad=True)
    y0_fine = torch.rand(1, 1, h_fine, w_fine, requires_grad=True)
    y_carrier = torch.rand(1, 1, h_carrier, w_carrier, requires_grad=True)
    y0_carrier = torch.rand(1, 1, h_carrier, w_carrier, requires_grad=True)

    target_fine = torch.rand(1, 1, h_fine, w_fine)
    regions = build_multiscale_regions(h_fine, w_fine, output_stride=2, region_sizes_px=(16, 32, 64), overlap=0.5)
    b_reg = regional_sum(target_fine, regions.boxes)
    disp = torch.full_like(b_reg, 50.0)

    outputs = {
        "y": y_fine,
        "y0": y0_fine,
        "y_carrier": y_carrier,
        "y0_carrier": y0_carrier,
        "regions": regions,
        "b_region": b_reg,
        "region_dispersion": disp,
    }

    # Case A: lambda_carrier_cell = 0.50, lambda_fine_cell = 0.25, lambda_cell = 0.50
    cfg_dual = RMRv3LossConfig(
        output_stride=2,
        lambda_count=0.0,
        lambda_flat_dm16=0.0,
        lambda_region_nb=0.0,
        lambda_cell=0.50,
        lambda_carrier_cell=0.50,
        lambda_fine_cell=0.25,
    )

    losses = compute_rmr_v3_losses(outputs, target_fine, cfg_dual)
    assert "cell_carrier" in losses
    assert "cell_fine" in losses

    # Total must equal exactly 0.25 * cell_fine + 0.50 * cell_carrier
    expected_total = 0.25 * losses["cell_fine"] + 0.50 * losses["cell_carrier"]
    assert torch.allclose(losses["total"], expected_total, atol=1e-5), (
        f"Total loss ({losses['total'].item()}) does not match single-counted fine + carrier "
        f"({expected_total.item()}). Fine loss was double-counted!"
    )

    # Gradients must flow to both fine and carrier
    losses["total"].backward()
    assert y_fine.grad is not None and torch.isfinite(y_fine.grad).all()
    assert y_carrier.grad is not None and torch.isfinite(y_carrier.grad).all()


# =========================================================================
# Test 2: Dual-Lattice Stride-2 with Non-Power-2 Regions & Odd Dimensions
# =========================================================================

def test_dual_lattice_odd_and_non_power2_regions_no_crash():
    """Verify subpixel_stride2 model handles arbitrary odd dims and non-power-2 regions without crash."""
    cfg = RMRv3Config(
        subpixel_stride2=True,
        output_stride=2,
        enable_solver=True,
        iterations=2,
        region_sizes_px=((10, 10), (20, 20), (30, 30)),
        region_overlap=0.5,
        adjoint_mode="radon_nikodym",
        use_barzilai_borwein=True,
    )
    model = RMRv3(cfg)
    model.eval()

    # Odd spatial dimensions that previously crashed due to M_feat != M_solver
    x = torch.randn(1, 3, 409, 301)
    out = model(x)

    assert out.y.shape[-2:] == (205, 151)  # (409+1)//2, (301+1)//2
    assert out.y_carrier.shape[-2:] == (103, 76)  # (205+1)//2, (151+1)//2

    # Verify region box count coherence
    m_boxes = out.regions.boxes.shape[0]
    m_evidence = out.b_region.shape[-1]
    assert m_boxes == m_evidence, f"Box count mismatch: {m_boxes} != {m_evidence}"

    # Verify SIRT solver executed cleanly with iterates
    assert len(out.iterates) == 3
    assert torch.isfinite(out.y).all()


# =========================================================================
# Test 3: Knowledge Distillation Loss Keys Schema Validation
# =========================================================================

def test_kd_loss_keys_schema_validation():
    """Verify configs with KD parameters pass schema validation without unknown key errors."""
    kd_config_path = Path("configs/rmr_v8/rmr_v8_with_kd.yaml")
    if kd_config_path.exists():
        raw_cfg = yaml.safe_load(kd_config_path.read_text(encoding="utf-8"))
        validate_v3_config(raw_cfg)

    # Test negative validation
    dummy_cfg = {
        "model": {"output_stride": 4},
        "loss": {"lambda_kd_spatial": -0.5},
    }
    with pytest.raises(ValueError, match="lambda_kd_spatial must be non-negative"):
        validate_v3_config(dummy_cfg)

    dummy_cfg2 = {
        "model": {"output_stride": 4},
        "loss": {"lambda_kd_count": -0.1},
    }
    with pytest.raises(ValueError, match="lambda_kd_count must be non-negative"):
        validate_v3_config(dummy_cfg2)


# =========================================================================
# Test 4: Tiled Prediction Metrics in Evaluation Summary
# =========================================================================

def test_evaluation_summary_tiled_metrics_present():
    """Verify summary.json includes tiled MAE and RMSE metrics when run_tiling=True."""
    from torch.utils.data import Dataset, DataLoader

    class SyntheticDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            h, w = 128, 128
            img = torch.rand(3, h, w)
            target = torch.zeros(1, h // 4, w // 4)
            target[0, 5, 5] = 2.0
            return {
                "image": img,
                "target_y": target,
                "id": f"syn_{idx}",
                "points": torch.tensor([[20.0, 20.0], [22.0, 22.0]]),
                "height": h,
                "width": w,
            }

    ds = SyntheticDataset()
    loader = DataLoader(ds, batch_size=1, collate_fn=lambda b: b)

    cfg = RMRv3Config(enable_solver=False)
    model = RMRv3(cfg)
    model.eval()

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        output_stride=4,
        run_tiling=True,
        tile_size=64,
        practical_halo=16,
    )

    # Verify tiled metrics are present in summary
    assert "mae_tiled_h0" in summary, "mae_tiled_h0 missing from summary"
    assert "rmse_tiled_h0" in summary, "rmse_tiled_h0 missing from summary"
    assert "mae_tiled_practical" in summary, "mae_tiled_practical missing from summary"
    assert "rmse_tiled_practical" in summary, "rmse_tiled_practical missing from summary"

    assert isinstance(summary["mae_tiled_practical"], float)
    assert math.isfinite(summary["mae_tiled_practical"])


# =========================================================================
# Test 5: DiAG Scale Routing Multi-Dtype Support & Mathematical Invariants
# =========================================================================

def test_diag_multi_dtype_and_invariants():
    """Verify DiAGScaleRoutingHead handles multiple dtypes and satisfies partition of unity."""
    from rmr_v3.model.perspective_geometry import DiAGScaleRoutingHead

    router = DiAGScaleRoutingHead(in_channels=32, num_scales=3)

    # 1. Zero-init check: uniform partition of unity
    x_f32 = torch.randn(2, 32, 16, 16, dtype=torch.float32)
    out_f32 = router(x_f32)
    assert out_f32.shape == (2, 3, 16, 16)
    assert torch.allclose(out_f32.sum(dim=1), torch.ones(2, 16, 16), atol=1e-5)

    # 2. Float64 input check
    router_f64 = DiAGScaleRoutingHead(in_channels=32, num_scales=3).to(dtype=torch.float64)
    x_f64 = torch.randn(2, 32, 16, 16, dtype=torch.float64)
    out_f64 = router_f64(x_f64)
    assert out_f64.dtype == torch.float64
    assert torch.allclose(out_f64.sum(dim=1), torch.ones(2, 16, 16, dtype=torch.float64), atol=1e-6)


# =========================================================================
# Test 6: Loss Section output_stride Validation
# =========================================================================

def test_output_stride_in_loss_section_validation():
    """Verify output_stride in loss: section is accepted and validated."""
    cfg_valid = {
        "model": {"output_stride": 4},
        "loss": {"output_stride": 4, "lambda_count": 1.0},
    }
    validate_v3_config(cfg_valid)

    cfg_valid_s2 = {
        "model": {"subpixel_stride2": True, "output_stride": 2},
        "loss": {"output_stride": 2, "lambda_count": 1.0},
    }
    validate_v3_config(cfg_valid_s2)

    cfg_invalid = {
        "model": {"output_stride": 4},
        "loss": {"output_stride": 3},
    }
    with pytest.raises(ValueError, match="output_stride in loss must be 2 or 4"):
        validate_v3_config(cfg_invalid)


# =========================================================================
# Test 7: scale_balanced_regional_nb_nll — empty-scale skip (BUG-7)
# =========================================================================

def test_scale_balanced_nb_nll_empty_scale_not_diluted():
    """BUG-7: scale_balanced_regional_nb_nll must not average empty scales as 0.0.

    Scenario: tiny 16×16 grid with region_sizes_px=(16, 32, 128). The 128px scale
    produces ZERO regions because the image is too small. Previously, the function
    appended 0.0 for the empty scale, which diluted the loss by 1/3. After the fix,
    the empty scale is skipped and the average is taken over the 2 active scales only.
    """
    from rmr_v3.losses.auxiliary import scale_balanced_regional_nb_nll

    # Build a region set where one scale will be empty (128px on a 16×16 grid)
    regions_full = build_multiscale_regions(
        height=16, width=16, output_stride=4,
        region_sizes_px=(16, 32, 128), include_full_image=False,
    )

    m = len(regions_full.boxes)
    if m == 0:
        pytest.skip("No regions generated; grid too small for any scale")

    # Identify which scales actually have regions
    active_sids = set(regions_full.scale_id.tolist())
    num_scales = int(regions_full.num_scales)

    # Compute loss with the (possibly sparse) region set
    target = torch.rand(2, 1, m).clamp_min(0.0)
    mean_pred = torch.rand(2, 1, m).clamp_min(1e-3) * 5.0
    disp = torch.full((2, 1, m), 50.0)

    loss_full = scale_balanced_regional_nb_nll(target, mean_pred, disp, regions_full)
    assert torch.isfinite(loss_full), "Loss must be finite"
    assert loss_full.item() >= 0.0, "NB NLL must be non-negative"

    # Now manually compute per-scale losses and verify scale_balanced averages only
    # over ACTIVE (non-empty) scales.
    from rmr_core.losses import negative_binomial_nll_mean_dispersion
    per_region = negative_binomial_nll_mean_dispersion(target, mean_pred, disp, reduction="none")
    manual_losses = []
    for sid in range(num_scales):
        mask = (regions_full.scale_id == sid)
        if not mask.any():
            continue  # skip empty — this is the fixed behavior
        count = mask.sum().clamp_min(1)
        manual_losses.append((per_region[..., mask].sum(dim=-1) / count).mean())

    if manual_losses:
        expected = torch.stack(manual_losses).mean()
        assert torch.allclose(loss_full, expected, atol=1e-5), (
            f"scale_balanced_regional_nb_nll gave {loss_full:.6f} but "
            f"manual (skip-empty) gave {expected:.6f}"
        )

    # Regression check: if an empty scale existed, the OLD (diluted) result would be smaller
    empty_sids = [sid for sid in range(num_scales) if not (regions_full.scale_id == sid).any()]
    if empty_sids:
        # Construct what the old code would have computed (with 0.0 for empty scales)
        old_losses = []
        for sid in range(num_scales):
            mask = (regions_full.scale_id == sid)
            count = mask.sum().clamp_min(1)
            old_losses.append((per_region[..., mask].sum(dim=-1) / count).mean())
        old_result = torch.stack(old_losses).mean()
        # The fixed result should be LARGER than the old diluted result
        assert loss_full.item() > old_result.item() - 1e-7, (
            "Fixed loss should not be smaller than old diluted result when empty scales exist"
        )


# =========================================================================
# Test 8: BUG-10 Truncated NB NLL Vectorization & Sample Isolation
# =========================================================================

def test_bug10_truncated_nb_nll_vectorization_and_sample_isolation():
    """Verify truncated_nb_nll_loss runs vectorized without host sync and preserves sample isolation."""
    from rmr_v3.losses.auxiliary import truncated_nb_nll_loss

    # B=2: Sample 0 has occupied regions, Sample 1 has ZERO occupied regions
    target = torch.tensor([[[5.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]]], requires_grad=False)
    mu = torch.tensor([[[4.0, 1.0, 3.0]], [[0.5, 0.5, 0.5]]], requires_grad=True)
    disp = torch.full_like(mu, 50.0)

    loss = truncated_nb_nll_loss(mu, disp, target)
    assert torch.isfinite(loss) and loss.item() > 0.0

    loss.backward()
    # Sample 1 has no occupied regions, so its gradients must be identically zero
    assert torch.all(mu.grad[1] == 0.0), "Sample 1 received non-zero gradient despite 0 occupied regions!"
    assert torch.any(mu.grad[0] != 0.0), "Sample 0 must receive gradients on occupied regions"

    # All-empty batch: must return 0.0 without crashing
    target_empty = torch.zeros_like(target)
    loss_empty = truncated_nb_nll_loss(mu, disp, target_empty)
    assert loss_empty.item() == 0.0


# =========================================================================
# Test 9: BUG-11 Top-K Hard Background Mining Sample Isolation
# =========================================================================

def test_bug11_topk_hard_background_sample_isolation():
    """Verify topk_hard_background_loss mines top-k per image independently without batch cross-talk."""
    from rmr_v3.losses.auxiliary import topk_hard_background_loss

    # B=2: Image 0 has huge background error, Image 1 has moderate background error
    y = torch.tensor([[[[10.0, 10.0], [0.0, 0.0]]], [[[2.0, 0.0], [0.0, 0.0]]]], requires_grad=True)
    target = torch.zeros_like(y)

    loss = topk_hard_background_loss(y, target, ratio=0.50, bg_threshold=1e-5)
    assert torch.isfinite(loss)

    loss.backward()
    # Both Image 0 and Image 1 must receive gradients because mining is per-sample
    assert torch.any(y.grad[0] > 0.0), "Image 0 must receive gradients"
    assert torch.any(y.grad[1] > 0.0), "Image 1 must receive gradients and not be starved by Image 0!"


# =========================================================================
# Test 10: BUG-12 Loss Operators Zero-Host-Sync on Empty Gates
# =========================================================================

def test_bug12_loss_operators_zero_host_sync_on_empty_gates():
    """Verify curvature_power_loss and physical_scale_alignment_loss work with zero host sync."""
    from rmr_v3.losses.auxiliary import curvature_power_loss, physical_scale_alignment_loss

    # 1. Curvature power loss with gate that selects nothing (threshold=100.0 on zero targets)
    y = torch.ones(1, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 1, 16, 16)
    loss_curv = curvature_power_loss(y, target, threshold=100.0, mode="hard")
    assert loss_curv.item() == 0.0

    # 2. Scale alignment loss on empty background
    scale_w = torch.full((1, 3, 16, 16), 1.0 / 3.0, requires_grad=True)
    loss_align = physical_scale_alignment_loss(scale_w, target, tau_sparse=1.0, mask_background=True)
    assert loss_align.item() == 0.0


# =========================================================================
# Test 11: BUG-13 Regional Evidence Head Full-Image Region Support
# =========================================================================

def test_bug13_regional_head_full_image_region_support():
    """Verify ProbabilisticRegionalEvidenceHead handles RegionSet with scale_id == -1 without crash."""
    from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead

    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32, hidden=48, region_sizes_px=(32, 64), hurdle_head=True
    )
    regions = build_multiscale_regions(64, 64, 4, (32, 64), 0.5, include_full_image=True, device="cpu")
    assert (regions.scale_id == -1).any(), "RegionSet must contain full-image scale -1"

    p4 = torch.randn(2, 32, 16, 16)
    p8 = torch.randn(2, 32, 8, 8)
    p16 = torch.randn(2, 32, 4, 4)

    out = head((p4, p8, p16), regions)
    m = regions.boxes.shape[0]
    assert out["mu_count"].shape == (2, 1, m), f"Expected [2, 1, {m}], got {out['mu_count'].shape}"
    assert out["dispersion"].shape == (2, 1, m)
    assert out["hurdle_logit"].shape == (2, 1, m)
    assert torch.isfinite(out["mu_count"]).all()


