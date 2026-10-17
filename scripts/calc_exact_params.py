import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import torch

from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.model.builder import build_neck_module

# Check existing necks
cfg_m1 = RMRv3Config.from_dict({
    "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
    "feature_width": 32,
    "neck_type": "aspp_lite",
    "output_stride": 4,
}, pretrained=False)
m1 = RMRv3(cfg_m1)
aspp_params = sum(p.numel() for p in m1.neck.parameters())
print(f"M1 aspp_lite neck params: {aspp_params:,}")

cfg_m2 = RMRv3Config.from_dict({
    "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
    "feature_width": 32,
    "neck_type": "rep_weighted",
    "output_stride": 4,
}, pretrained=False)
m2 = RMRv3(cfg_m2)
repdw_params = sum(p.numel() for p in m2.neck.parameters())
print(f"M2 rep_weighted neck params: {repdw_params:,}")

backbone_heads_params = sum(p.numel() for p in m1.parameters()) - aspp_params
print(f"Backbone + Heads params (W=32): {backbone_heads_params:,}")

# Now if neck is RepASPPNeck (16,192):
total_track_a = backbone_heads_params + 16192
print(f"Total Model with RepASPPNeck (W=32): {total_track_a:,} (Limit: 104,441)")

# How about Track B (W=64)?
cfg_m4 = RMRv3Config.from_dict({
    "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
    "feature_width": 64,
    "neck_type": "rep_weighted",
    "output_stride": 4,
}, pretrained=False)
m4 = RMRv3(cfg_m4)
total_m4 = sum(p.numel() for p in m4.parameters())
print(f"M4 Full Model (W=64, rep_weighted): {total_m4:,} (Limit: 200,000)")
