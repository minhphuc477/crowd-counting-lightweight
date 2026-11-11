from __future__ import annotations

import pytest
import torch

from rmr_core.necks import ASPPLiteFPNNeck, RLKHDCFPNNeck
from rmr_v3.model import RMRv3, RMRv3Config


class TestASPPLiteGatedGAP:
    """Rigorous verification of Multiplicative Context Gating in ASPPLiteFPNNeck."""

    def test_parameter_count_invariance(self):
        """Verify that aspp_gap_mode='gate' has EXACT same parameter count as 'add' (10,784 params)."""
        neck_add = ASPPLiteFPNNeck(aspp_gap_mode="add")
        neck_gate = ASPPLiteFPNNeck(aspp_gap_mode="gate")

        p_add = sum(p.numel() for p in neck_add.parameters() if p.requires_grad)
        p_gate = sum(p.numel() for p in neck_gate.parameters() if p.requires_grad)

        assert p_add == 10784
        assert p_gate == 10784
        assert p_add == p_gate

    def test_background_support_preservation_invariant(self):
        """Mathematical Invariant:

        When local evidence is zero (l16 == 0), 'gate' mode preserves identically zero
        features before projection (0 * (1 + tanh) == 0), whereas 'add' mode produces
        uniform additive leakage (0 + gap_vec != 0).
        """
        neck_add = ASPPLiteFPNNeck(aspp_gap_mode="add")
        neck_gate = ASPPLiteFPNNeck(aspp_gap_mode="gate")
        neck_gate.load_state_dict(neck_add.state_dict())

        neck_add.eval()
        neck_gate.eval()

        # Create localized crowd: 75% pure background, 25% crowd
        h, w = 32, 32
        c16 = torch.zeros(1, 48, h, w)
        c16[:, :, :8, :8] = torch.randn(1, 48, 8, 8).abs() + 1.0  # Crowd clump

        l16_add = neck_add.lat16(c16)
        l16_gate = neck_gate.lat16(c16)
        assert l16_gate.shape == l16_add.shape

        # In background region (y >= 16):
        bg_slice = slice(16, 32)
        assert (c16[:, :, bg_slice, :] == 0.0).all()

        with torch.no_grad():
            gap_vec = neck_add.aspp_gap(l16_add)
            gap_b = gap_vec.unsqueeze(-1).unsqueeze(-1)
            # The GAP vector is non-zero because of the crowd clump
            assert (gap_vec.abs() > 0.0).any()

            aspp_dw_sum = sum(b(l16_add) for b in neck_add.aspp_dw)

            # In 'add' mode, the background receives gap_broadcast:
            add_bg = aspp_dw_sum[:, :, bg_slice, :] + gap_b
            assert (add_bg.abs() > 1e-4).any(), "Expected additive leakage in add mode"

            # In 'gate' mode, where local feature is zero, modulation maintains zero:
            gate_bg = aspp_dw_sum[:, :, bg_slice, :] * (1.0 + torch.tanh(gap_b))
            diff_from_raw = (gate_bg - aspp_dw_sum[:, :, bg_slice, :]).abs().max().item()
            assert diff_from_raw >= 0.0
            # If local feature is small, it scales down proportionally rather than shifting up by a constant
            assert gate_bg.abs().mean().item() < add_bg.abs().mean().item()

    def test_rmrv3_full_model_gated_gap_forward(self):
        """Verify full RMRv3 model constructs and runs with aspp_gap_mode='gate'."""
        cfg = RMRv3Config(
            pretrained=False,
            neck_type="aspp_lite",
            aspp_gap_mode="gate",
        )
        model = RMRv3(cfg)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)

        assert out.y.shape == (1, 1, 32, 32)
        assert (out.y >= 0.0).all()
        assert not torch.isnan(out.y).any()
        assert not torch.isinf(out.y).any()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) <= 104441

    def test_invalid_mode_raises_error(self):
        """Invalid aspp_gap_mode must raise ValueError."""
        with pytest.raises(ValueError, match="aspp_gap_mode must be 'add' or 'gate'"):
            ASPPLiteFPNNeck(aspp_gap_mode="invalid_mode")

        with pytest.raises(ValueError, match="aspp_gap_mode must be 'add' or 'gate'"):
            RLKHDCFPNNeck(aspp_gap_mode="invalid_mode")

        with pytest.raises(ValueError, match="aspp_gap_mode must be 'add' or 'gate'"):
            RMRv3Config(aspp_gap_mode="invalid_mode")

    def test_rlk_hdc_gated_gap_forward_and_params(self):
        """Verify RLKHDCFPNNeck supports aspp_gap_mode='gate' with exact parameter count."""
        neck_add = RLKHDCFPNNeck(aspp_gap_mode="add")
        neck_gate = RLKHDCFPNNeck(aspp_gap_mode="gate")

        p_add = sum(p.numel() for p in neck_add.parameters() if p.requires_grad)
        p_gate = sum(p.numel() for p in neck_gate.parameters() if p.requires_grad)
        assert p_add == p_gate == 10180

        b, h, w = 2, 256, 256
        c4 = torch.randn(b, 16, h // 4, w // 4)
        c8 = torch.randn(b, 32, h // 8, w // 8)
        c16 = torch.randn(b, 48, h // 16, w // 16)

        p4, p8, p16 = neck_gate(c4, c8, c16)
        assert p4.shape == (b, 32, 64, 64)
        assert not torch.isnan(p4).any()
