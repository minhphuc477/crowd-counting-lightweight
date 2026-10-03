import sys
from pathlib import Path
sys.path.insert(0, ".")

import yaml
from rmr_v3.config.validator import validate_v3_config
from rmr_v3.model import RMRv3, RMRv3Config

configs = [
    "configs/rmr_research/sub60_e50_hdc_lite_neck.yaml",
    "configs/rmr_research/sub60_e51_ci_cell_dynamic_range.yaml",
    "configs/rmr_research/sub60_e52_decoupled_carrier_supervision.yaml",
    "configs/rmr_research/sub60_e53_hdc_ci_cell_synthesis.yaml"
]

for cp in configs:
    with open(cp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_v3_config(cfg)
    m_cfg = RMRv3Config.from_dict(cfg["model"])
    model = RMRv3(m_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name = Path(cp).name
    assert n_params <= 104441, f"Parameter budget exceeded: {n_params} > 104441"
    print(f"{name:<45} : PASS | Trainable Params = {n_params} (<= 104,441)")
print("ALL NEW CONFIGS VALIDATED SUCCESSFULLY!")
