import pytest
from pathlib import Path
import yaml
import torch

from rmr_v3.engine import make_model, make_loss_cfg
from rmr_v3.losses import compute_rmr_v3_losses

V54_CONFIGS = [
    "configs/rmr_v54/v54_track_a_control_sirt.yaml",
    "configs/rmr_v54/v54_track_a_trust_damped_rn.yaml",
    "configs/rmr_v54/v54_track_a_raw_rn_ablation.yaml",
    "configs/rmr_v54/v54_track_b_control_sirt.yaml",
    "configs/rmr_v54/v54_track_b_trust_damped_rn.yaml",
]

@pytest.mark.parametrize("cfg_path", V54_CONFIGS)
def test_v54_config_contract_and_forward_backward(cfg_path: str):
    path = Path(cfg_path)
    assert path.exists(), f"Config file not found: {cfg_path}"
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model, uniform_rel = make_model(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    limit = cfg["model"]["max_trainable_params"]
    assert total_params <= limit, f"{path.name} parameter violation: {total_params} > {limit}"
    
    # Standalone 0.0% KD check
    assert "teacher" not in cfg
    assert "distillation" not in cfg
    
    # Forward check
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x, uniform_reliability=uniform_rel)
    assert "y" in out
    
    # Backward check
    model.train()
    loss_cfg = make_loss_cfg(cfg)
    stride = int(cfg["model"]["output_stride"])
    target_h, target_w = 256 // stride, 256 // stride
    target_y = torch.zeros(2, 1, target_h, target_w)
    target_y[0, 0, 5, 5] = 1.0
    target_y[1, 0, 8, 8] = 2.0
    pts1 = torch.tensor([[20.0, 20.0]])
    pts2 = torch.tensor([[32.0, 32.0], [35.0, 35.0]])
    
    out_train = model(x, uniform_reliability=uniform_rel)
    losses = compute_rmr_v3_losses(
        outputs=out_train,
        target_y=target_y,
        cfg=loss_cfg,
        points=[pts1, pts2],
    )
    assert "total" in losses
    loss = losses["total"]
    loss.backward()
