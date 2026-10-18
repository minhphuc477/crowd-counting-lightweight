from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.engine import make_model, make_loss_cfg
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_core.data import rasterize_points


ABLATION_CONFIGS = [
    ("configs/rmr_v50/v50_track_a_repaspp.yaml", 104441, 4),
    ("configs/rmr_v50/v50_track_b_nextgen.yaml", 200000, 2),
    ("configs/rmr_v50/ablations/v50_abl_no_solver.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_no_evidence.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_standard_nb.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_hurdle_product.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_hurdle_occupancy.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_no_native_pooling.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_coupled_trust.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_no_morozov.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_neck_additive_fpn.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_solver_t2.yaml", 104441, 4),
    ("configs/rmr_v50/ablations/v50_abl_track_b_mean_only.yaml", 200000, 2),
]


@pytest.mark.parametrize("cfg_path_str,max_params,stride", ABLATION_CONFIGS)
def test_ablation_config_validity_and_forward(cfg_path_str: str, max_params: int, stride: int) -> None:
    cfg_path = Path(cfg_path_str)
    assert cfg_path.exists(), f"Config file {cfg_path_str} does not exist"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    model, unif = make_model(raw_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= max_params, f"{cfg_path_str} exceeded budget: {total_params} > {max_params}"

    # Fast forward pass on batch 1
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    expected_dim = 256 // stride
    assert out.y.shape[-2:] == (expected_dim, expected_dim), f"Expected shape ({expected_dim}, {expected_dim}), got {out.y.shape}"
