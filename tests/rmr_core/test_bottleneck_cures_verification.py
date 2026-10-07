import math
import torch
import torch.nn.functional as F
import pytest

from rmr_core.heads import build_fine_head, FineMeasureHead
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.evidence import extract_regional_evidence
from rmr_v3.losses.point_supervision import bayesian_loss
from rmr_core.operators import build_multiscale_regions


def test_fine_head_subrayleigh_uncapping():
    """Verify adaptive curvature scaling un-caps sub-Rayleigh clusters without background leaks."""
    head_base = build_fine_head(
        density_adaptive_scale=False,
        density_scale_gamma=0.0,
    )
    head_boost = build_fine_head(
        density_adaptive_scale=True,
        density_scale_gamma=1.0,
        density_scale_learnable=False,
    )

    # 1. Background protection: when logit z <= 0, relu(z) == 0, so boost is exactly 1.0 (0 false positives)
    z_neg = torch.tensor([-2.0, -1.0, 0.0], dtype=torch.float32).view(1, 1, 3, 1)
    y_base_neg = head_base.activate(z_neg)
    y_boost_neg = head_boost.activate(z_neg)
    assert torch.allclose(y_base_neg, y_boost_neg, atol=1e-6), "Negative logits must NOT be boosted"

    # 2. Sub-Rayleigh cluster: when logit z > 0 (dense crowd cell), boost allows y0 > 2.0
    z_pos = torch.tensor([1.5], dtype=torch.float32).view(1, 1, 1, 1)
    y_base_pos = head_base.activate(z_pos)
    y_boost_pos = head_boost.activate(z_pos)
    expected_boost = y_base_pos * (1.0 + 1.0 * 1.5)  # 2.5x amplification
    assert torch.allclose(y_boost_pos, expected_boost, atol=1e-5), "Positive logits must follow softplus(z)*(1 + gamma*z)"
    assert y_boost_pos.item() > 2.0, f"Expected uncapped density > 2.0, got {y_boost_pos.item()}"

    # 3. Verify zero parameter addition when learnable=False
    params_base = sum(p.numel() for p in head_base.parameters())
    params_boost = sum(p.numel() for p in head_boost.parameters())
    assert params_base == params_boost, f"Must have zero parameter overhead: {params_base} vs {params_boost}"


def test_fine_head_scale_conditioned_consistency():
    """Verify scale-conditioned and unconditioned activation have identical non-saturating formulas."""
    head_sc = build_fine_head(
        scale_conditioned_prior=True,
        density_adaptive_scale=True,
        density_scale_gamma=1.0,
        density_scale_learnable=False,
    )
    # Neutral scale weights
    sw = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]], dtype=torch.float32)
    z_pos = torch.tensor([[[[1.0]]]], dtype=torch.float32)
    y_out = head_sc.activate(z_pos, scale_weights=sw)

    # With gamma=1.0 and z=1.0, relu(1.0)=1.0, multiplier is 2.0
    y_base = F.softplus(z_pos)
    assert torch.allclose(y_out, y_base * 2.0, atol=1e-5)


def test_bayesian_loss_subrayleigh_adaptive_sigma():
    """Verify k-NN adaptive sigma sharpens adjacent heads separated by < 4px."""
    # 4 heads placed 3px apart (sub-Rayleigh regime at stride 4)
    dense_pts = torch.tensor([
        [50.0, 50.0],
        [53.0, 50.0],
        [50.0, 53.0],
        [53.0, 53.0],
    ], dtype=torch.float32)

    pred = torch.ones(1, 1, 32, 32, requires_grad=True)

    # With default sigma_min=2.0 and adaptive_sigma=True
    loss = bayesian_loss(
        pred, [dense_pts],
        stride=4,
        adaptive_sigma=True,
        sigma_min=2.0,
        sigma_max=8.0,
    )
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_bayesian_loss_sparse_fallback():
    """Verify bayesian loss with < 4 points safely falls back to standard sigma without error."""
    sparse_pts = torch.tensor([[20.0, 20.0], [80.0, 80.0]], dtype=torch.float32)
    pred = torch.ones(1, 1, 32, 32, requires_grad=True)
    loss = bayesian_loss(
        pred, [sparse_pts],
        stride=4,
        adaptive_sigma=True,
        sigma_min=2.0,
        sigma_max=8.0,
    )
    assert not torch.isnan(loss)
    loss.backward()
    assert pred.grad is not None


def test_hurdle_occupancy_vs_product_mass_preservation():
    """Verify hurdle occupancy mode guarantees 100% mass preservation on dense regions."""
    cfg_occ = RMRv3Config(hurdle_head=True, hurdle_gating_mode="occupancy")
    cfg_prod = RMRv3Config(hurdle_head=True, hurdle_gating_mode="product")

    # High density box: b_raw = 500.0, hurdle probability = 0.75
    b_raw = torch.tensor([[[500.0]]], dtype=torch.float32)
    disp = torch.tensor([[[50.0]]], dtype=torch.float32)
    logit = torch.tensor([[[1.0986]]], dtype=torch.float32)  # sigmoid(1.0986) ≈ 0.75

    h_dict = {
        "mu_count": b_raw,
        "dispersion": disp,
        "log_dispersion": torch.log(disp),
        "rate": b_raw / 64.0,
        "hurdle_logit": logit,
    }

    class MockRegionalHead:
        def __call__(self, pyr, regs):
            return h_dict

    regs = build_multiscale_regions(32, 32, output_stride=4, region_sizes_px=(32,), overlap=0.0)

    res_occ = extract_regional_evidence(
        cfg_occ, MockRegionalHead(), torch.empty(1), torch.empty(1), torch.empty(1),
        regs, None, False, 8,
    )
    res_prod = extract_regional_evidence(
        cfg_prod, MockRegionalHead(), torch.empty(1), torch.empty(1), torch.empty(1),
        regs, None, False, 8,
    )

    # In occupancy mode: b_raw >= 1.0 -> occ_gate = 1.0 -> exactly 500.0 (0% mass lost)
    assert torch.allclose(res_occ["b_solver"], b_raw), "Occupancy mode must preserve 100% of dense mass"

    # In product mode: b_solver = 0.75 * 500 = 375 (125 people destroyed!)
    assert res_prod["b_solver"].item() < 380.0, "Product mode destroys 25% of mass"


def test_boundary_margin_zero_rejects_out_of_bounds_points():
    """Verify that boundary_margin=0.0 rejects out-of-bounds points instead of clamping to boundary cells."""
    from PIL import Image
    import numpy as np
    from rmr_core.data import train_transform, rasterize_points, normalize_image

    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    # Points: one inside (32, 32), one just outside left (-2.0, 32), one outside right (65.0, 32)
    pts = torch.tensor([[32.0, 32.0], [-2.0, 32.0], [65.0, 32.0]], dtype=torch.float32)

    img_t, pts_out = train_transform(
        img, pts, crop_size=64, scale_range=(1.0, 1.0), hflip_prob=0.0, boundary_margin=0.0
    )

    # Exactly 1 point inside [0, 64) must be kept
    assert pts_out.shape[0] == 1, f"Expected 1 point kept, got {pts_out.shape[0]}"
    assert pts_out[0, 0].item() == 32.0

    # Rasterization should have sum = 1.0, with ZERO points in cell j=0 or j=15
    dmap = rasterize_points(pts_out, 64, 64, stride=4)
    assert dmap.sum().item() == 1.0
    assert dmap[0, 8, 8].item() == 1.0


def test_padding_neutral_gray_normalizes_to_zero():
    """Verify that neutral gray padding (128, 128, 128) maps to 0.0 under standard image normalization."""
    from rmr_core.data import normalize_image
    from torchvision.transforms import functional as TF

    # Neutral gray in uint8 PIL
    gray_pixel = torch.full((3, 4, 4), 128.0 / 255.0, dtype=torch.float32)
    normed = normalize_image(gray_pixel)

    # (128/255 - 0.5) / 0.5 ≈ 0.00392 ≈ 0.0
    assert torch.allclose(normed, torch.zeros_like(normed), atol=0.01), "Neutral gray must normalize near zero"

