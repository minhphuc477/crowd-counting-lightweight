import torch
import torch.nn as nn
from rmr_core.necks import RepDWBlock7x7, RepWeightedFPNNeck


def test_rep_dw_block_7x7_deploy_equivalence():
    torch.manual_seed(42)
    block = RepDWBlock7x7(channels=32)
    block.eval()

    x = torch.randn(2, 32, 64, 64)
    out_train = block(x)

    block.switch_to_deploy()
    assert block.is_deployed
    out_deploy = block(x)

    max_diff = (out_train - out_deploy).abs().max().item()
    assert max_diff < 1e-4, f"Deploy output mismatch: {max_diff}"


def test_rep_weighted_fpn_neck_forward_and_backward():
    torch.manual_seed(42)
    neck = RepWeightedFPNNeck(in_channels=(16, 32, 64), width=32)
    neck.train()

    c4 = torch.randn(2, 16, 128, 128)
    c8 = torch.randn(2, 32, 64, 64)
    c16 = torch.randn(2, 64, 32, 32)

    p4, p8, p16 = neck(c4, c8, c16)

    assert p4.shape == (2, 32, 128, 128)
    assert p8.shape == (2, 32, 64, 64)
    assert p16.shape == (2, 32, 32, 32)

    loss = p4.sum() + p8.sum() + p16.sum()
    loss.backward()

    assert neck.weight_16.grad is not None
    assert neck.weight_8.grad is not None
    assert neck.weight_4.grad is not None


def test_rep_weighted_fpn_neck_deploy():
    torch.manual_seed(42)
    neck = RepWeightedFPNNeck(in_channels=(16, 32, 48), width=32)
    neck.eval()

    c4 = torch.randn(1, 16, 32, 32)
    c8 = torch.randn(1, 32, 16, 16)
    c16 = torch.randn(1, 48, 8, 8)

    p4_pre, p8_pre, p16_pre = neck(c4, c8, c16)
    neck.switch_to_deploy()
    p4_post, p8_post, p16_post = neck(c4, c8, c16)

    assert (p4_pre - p4_post).abs().max().item() < 1e-4
    assert (p8_pre - p8_post).abs().max().item() < 1e-4
    assert (p16_pre - p16_post).abs().max().item() < 1e-4
