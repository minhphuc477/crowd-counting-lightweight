import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
import torch
from rmr_v3.config import validate_v3_config
from rmr_v3.train import make_model, make_loss_cfg

def audit_configs():
    config_dir = REPO_ROOT / "configs" / "rmr_v9"
    configs = sorted(config_dir.glob("*.yaml"))
    assert len(configs) == 6, f"Expected 6 configs, found {len(configs)}"
    
    print(f"=== Auditing {len(configs)} RMR-v9 Configurations ===")
    for p in configs:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # 1. Validation
        validate_v3_config(cfg)
        
        # 2. Model instantiation
        model, uniform = make_model(cfg)
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in model.parameters())
        
        assert trainable_params <= 105_000, f"{p.name} trainable params {trainable_params} > 105,000"
        
        # 3. Loss config
        loss_cfg = make_loss_cfg(cfg)
        
        # 4. Learning rates and schedules
        train_cfg = cfg.get("train", {})
        lr = float(train_cfg.get("lr", 0.0))
        assert lr > 0.0, f"Invalid lr {lr} in {p.name}"
        assert train_cfg.get("epochs", 0) > 0, f"Invalid epochs in {p.name}"
        assert train_cfg.get("batch_size", 0) > 0, f"Invalid batch_size in {p.name}"
        
        # 5. Data manifests
        data_cfg = cfg.get("data", {})
        train_man = data_cfg.get("train_manifest")
        val_man = data_cfg.get("val_manifest")
        assert train_man and (REPO_ROOT / train_man).exists(), f"Train manifest not found: {train_man}"
        assert val_man and (REPO_ROOT / val_man).exists(), f"Val manifest not found: {val_man}"
        
        print(f"PASS: {p.name}")
        print(f"      Trainable Params: {trainable_params:,} | Total Params: {total_params:,}")
        print(f"      Uniform: {uniform} | LR: {lr} | Backbone LR Scale: {train_cfg.get('backbone_lr_scale')}")
        print(f"      Solver: {model.cfg.enable_solver} (mode={model.cfg.solver_mode}, iters={model.cfg.iterations}, tau={model.cfg.proximal_tau})")
        print(f"      Loss: count_mode={loss_cfg.count_loss_mode}, dm={loss_cfg.allocation_loss_type}, kappa={loss_cfg.kappa_flat16}")
        print(f"      Data: train={train_man}, val={val_man}\n")

if __name__ == "__main__":
    audit_configs()
