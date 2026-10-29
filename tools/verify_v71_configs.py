import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
from rmr_v3.engine import make_model

def main():
    configs = sorted(Path("configs/rmr_v71").glob("*.yaml"))
    print(f"Verifying {len(configs)} configs in configs/rmr_v71/...")
    for p in configs:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model, _ = make_model(cfg)
        total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        ceiling = cfg["model"]["max_trainable_params"]
        assert total_params <= ceiling, f"Exceeded ceiling: {total_params} > {ceiling}"
        
        # Test synthetic forward pass
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        print(f"PASSED: {p.name} | Trainable params: {total_params:,} (<= {ceiling:,}) | y shape: {tuple(out.y.shape)}")

if __name__ == "__main__":
    main()
