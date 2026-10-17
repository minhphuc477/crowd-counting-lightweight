from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_core.data import rasterize_points
from rmr_v3.engine import make_loss_cfg, make_model
from rmr_v3.losses import compute_rmr_v3_losses


def test_v50_track_a_config_and_forward() -> None:
    cfg_path = Path("configs/rmr_v50/v50_track_a_repaspp.yaml")
    assert cfg_path.exists(), "v50_track_a_repaspp.yaml must exist"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    model, unif = make_model(raw_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_p = raw_cfg["model"]["max_trainable_params"]

    # Parameter budget constraint
    assert total_params <= max_p, f"Track A exceeded parameter budget: {total_params} > {max_p}"
    assert total_params <= 104441, f"Track A exceeded strict 104,441 budget: {total_params}"

    # Forward pass
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.y.shape == (2, 1, 128, 128)
    assert out.y0.shape == (2, 1, 128, 128)

    # Loss computation
    pts1 = torch.tensor([[100.0, 100.0], [200.0, 250.0]])
    pts2 = torch.tensor([[50.0, 50.0]])
    target_y = torch.stack([
        rasterize_points(pts1, 512, 512, stride=4),
        rasterize_points(pts2, 512, 512, stride=4),
    ])
    loss_cfg = make_loss_cfg(raw_cfg)
    losses = compute_rmr_v3_losses(dict(out), target_y, loss_cfg, points=[pts1, pts2])

    assert losses["total"].item() > 0.0
    # Backward gradient flow
    losses["total"].backward()
    assert model.fusion.ref16.rbr_dense[0].weight.grad is not None
    assert model.fusion.ref16.rbr_dense[0].weight.grad.norm().item() > 0.0


def test_v50_track_b_config_and_forward() -> None:
    cfg_path = Path("configs/rmr_v50/v50_track_b_nextgen.yaml")
    assert cfg_path.exists(), "v50_track_b_nextgen.yaml must exist"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    model, unif = make_model(raw_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_p = raw_cfg["model"]["max_trainable_params"]

    # Parameter budget constraint
    assert total_params <= max_p, f"Track B exceeded parameter budget: {total_params} > {max_p}"
    assert total_params <= 200000, f"Track B exceeded strict 200,000 budget: {total_params}"

    # Forward pass on Subpixel Stride 2
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.y.shape == (2, 1, 256, 256)
    assert out.y0.shape == (2, 1, 256, 256)

    # Loss computation
    pts1 = torch.tensor([[100.0, 100.0], [200.0, 250.0]])
    pts2 = torch.tensor([[50.0, 50.0]])
    target_y = torch.stack([
        rasterize_points(pts1, 512, 512, stride=2),
        rasterize_points(pts2, 512, 512, stride=2),
    ])
    loss_cfg = make_loss_cfg(raw_cfg)
    losses = compute_rmr_v3_losses(dict(out), target_y, loss_cfg, points=[pts1, pts2])

    assert losses["total"].item() > 0.0
    losses["total"].backward()
    assert model.fusion.ref16.rbr_dense[0].weight.grad is not None
    assert model.fusion.ref16.rbr_dense[0].weight.grad.norm().item() > 0.0


def test_v50_purified_loss_orthogonality() -> None:
    raw_cfg = yaml.safe_load(Path("configs/rmr_v50/v50_track_a_repaspp.yaml").read_text(encoding="utf-8"))
    loss_cfg = make_loss_cfg(raw_cfg)

    # Core orthogonal losses must be active
    assert loss_cfg.lambda_count > 0.0
    assert loss_cfg.lambda_flat_dm16 > 0.0
    assert loss_cfg.lambda_region_nb > 0.0
    assert loss_cfg.faithful_regional_nb is True
    assert loss_cfg.regional_nb_beta == 0.5

    # Conflicting auxiliary losses must be strictly inactive
    assert loss_cfg.lambda_cell == 0.0
    assert loss_cfg.lambda_hurdle == 0.0
    assert loss_cfg.lambda_trunc_nb == 0.0
    assert loss_cfg.lambda_scale_align == 0.0
    assert loss_cfg.lambda_curvature == 0.0
    assert loss_cfg.lambda_hard_bg == 0.0
    assert loss_cfg.lambda_fg_gate == 0.0
