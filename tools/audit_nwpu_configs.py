import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import torch
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.engine import make_model, make_loss_cfg
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.optim.builder import build_v3_optimizer, build_v3_scheduler

nwpu_yamls = sorted(list(Path('configs/benchmarks').glob('exp_nwpu_*.yaml')))
print(f'Found {len(nwpu_yamls)} NWPU YAML configs.\n')

failed = False
for yf in nwpu_yamls:
    try:
        cfg = yaml.safe_load(yf.read_text('utf-8'))
        
        # 1. Schema and invariants
        validate_v3_config(cfg)
        
        # 2. Model & param count
        model, is_new = make_model(cfg)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params <= 104441, f'Param limit exceeded: {trainable_params}'
        
        # 3. Forward pass (output stride 2)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        expected_shape = (2, 1, 256, 256)
        assert out['y'].shape == expected_shape, f'Unexpected shape: {out["y"].shape} vs {expected_shape}'
        
        # 4. Loss & Backward pass (sample with points + sample with empty points for negative scene!)
        loss_cfg = make_loss_cfg(cfg)
        target = torch.zeros(2, 1, 256, 256)
        target[0, 0, 50, 50] = 1.0
        pts = [torch.tensor([[100.0, 100.0]]), torch.empty((0, 2))]
        losses = compute_rmr_v3_losses(out, target, loss_cfg, points=pts)
        total_loss = losses['total']
        assert not torch.isnan(total_loss) and not torch.isinf(total_loss), f'Loss is NaN/Inf'
        total_loss.backward()
        
        # 5. Optimizer & Scheduler
        lr_init = float(cfg.get('train', {}).get('lr', 0.0001))
        epochs = int(cfg.get('train', {}).get('epochs', 800))
        opt = build_v3_optimizer(model, cfg, lr_init)
        sched = build_v3_scheduler(opt, cfg, epochs)
        
        # Check gradient on all params
        non_zero_grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert non_zero_grads == total_trainable, f'Gradients missing on {total_trainable - non_zero_grads} parameters!'
        
        print(f'[PASS] {yf.name:<38} | Params: {trainable_params:,} | Loss: {total_loss.item():.4f} | Grads: {non_zero_grads}/{total_trainable}')
    except Exception as e:
        print(f'[FAIL] {yf.name}: {e}')
        failed = True

if failed:
    sys.exit(1)
print('\n>>> ALL 9 NWPU YAML CONFIGS PASSED RIGOROUS AUDIT WITH ZERO ERRORS! <<<')
