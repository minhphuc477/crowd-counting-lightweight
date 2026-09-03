"""Unit tests for MICF-v2 (Monotonic Integral Count Field).

Verifies:
1. Exact telescoping identity: sum(Delta_xy C) == C[..., -1, -1] == sum(Y).
2. Exact algebraic inversion: Delta_xy(cumsum_2d(Y)) == Y.
3. DirectionalIntegralContext: shapes, gradient flow, and prefix average correctness.
4. MICFLoss: validity penalty is zero for true fields, positive for invalid fields.
5. MICFLite forward and predict methods for both local and cumulative heads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
    IntegralLossOnLocalCount,
    MICFLoss,
)
from hpc.models.integral_context import DirectionalIntegralContext
from hpc.models.micf_lite import MICFLite


class TestMICFv2:
    def test_algebraic_inversion_and_telescoping(self):
        # Arbitrary positive discrete cell counts [1, 1, 8, 8]
        y = torch.tensor([
            [0.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 2.0, 1.0, 1.0],
            [4.0, 0.0, 0.0, 2.0],
        ]).view(1, 1, 4, 4)

        # Forward 2D cumulative sum
        c = cell_counts_to_cumulative_field(y, orientation="TL")

        # Check telescoping boundary identity: bottom-right corner == total count
        assert c[0, 0, -1, -1].item() == y.sum().item()

        # Invert via discrete mixed difference Delta_xy C
        y_recovered = discrete_mixed_difference(c)

        # Check exact reconstruction
        assert torch.allclose(y, y_recovered, atol=1e-6)

        # Check sum of recovered cell counts equals bottom-right corner
        assert abs(y_recovered.sum().item() - c[0, 0, -1, -1].item()) < 1e-6

    def test_orientations(self):
        y = torch.ones(1, 1, 4, 4)
        c_tl = cell_counts_to_cumulative_field(y, orientation="TL")
        c_tr = cell_counts_to_cumulative_field(y, orientation="TR")
        c_bl = cell_counts_to_cumulative_field(y, orientation="BL")
        c_br = cell_counts_to_cumulative_field(y, orientation="BR")

        assert c_tl[0, 0, -1, -1] == 16.0
        assert c_tr[0, 0, -1, 0] == 16.0
        assert c_bl[0, 0, 0, -1] == 16.0
        assert c_br[0, 0, 0, 0] == 16.0

    def test_directional_integral_context(self):
        module = DirectionalIntegralContext(channels=16, use_residual=True)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)

        out = module(x)
        assert out.shape == (2, 16, 8, 8), f"Expected (2,16,8,8), got {tuple(out.shape)}"

        # Check gradient flow
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Verify TL prefix average is correct for constant-1 input
        with torch.no_grad():
            from hpc.models.integral_context import (
                normalized_integral_tl, normalized_integral_tr,
                normalized_integral_bl, normalized_integral_br,
            )
            ones = torch.ones(1, 1, 4, 4)
            tl = normalized_integral_tl(ones)
            # Each cell should be 1.0 (sum of (i+1)(j+1) ones / (i+1)(j+1))
            assert torch.allclose(tl, torch.ones_like(tl), atol=1e-5)

            # TR: flip-normalize-flip; should also be all-ones for constant input
            tr = normalized_integral_tr(ones)
            assert torch.allclose(tr, torch.ones_like(tr), atol=1e-5)

            bl = normalized_integral_bl(ones)
            assert torch.allclose(bl, torch.ones_like(bl), atol=1e-5)

            br = normalized_integral_br(ones)
            assert torch.allclose(br, torch.ones_like(br), atol=1e-5)

    def test_micf_loss(self):
        crit = MICFLoss(field_loss="smooth_l1", lambda_valid=1.0)

        # Perfect prediction (identical to target)
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(1, 1, 2, 2)
        c = cell_counts_to_cumulative_field(y)
        loss, comp = crit(c, c, return_components=True)

        assert abs(loss.item()) < 1e-6
        assert comp["validity_loss"] == 0.0
        assert comp["violation_rate"] == 0.0

        # Invalid prediction: decreasing field (negative mixed difference)
        c_invalid = torch.tensor([[5.0, 2.0], [3.0, 1.0]]).view(1, 1, 2, 2)
        loss_inv, comp_inv = crit(c_invalid, c, return_components=True)

        assert comp_inv["validity_loss"] > 0.0
        assert comp_inv["violation_rate"] > 0.0

    def test_integral_loss_on_local_count(self):
        crit = IntegralLossOnLocalCount(loss_type="l1")
        y1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).view(1, 1, 2, 2)
        y2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).view(1, 1, 2, 2)

        # Cumsum of y1: [[1, 1], [1, 2]]
        # Cumsum of y2: [[0, 1], [1, 2]]
        # Diff: [[1, 0], [0, 0]] -> mean = 1/4 = 0.25
        loss = crit(y1, y2)
        assert abs(loss.item() - 0.25) < 1e-6

    def test_micf_lite_models(self):
        # 1. Cumulative head without context (B3/B4)
        m_cum = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="cumulative",
            use_integral_context=False,
            output_stride=16,
        )
        x = torch.randn(1, 3, 128, 128)
        field_cum = m_cum(x)
        assert field_cum.shape == (1, 1, 8, 8)
        count_cum, map_cum = m_cum.predict(x, pad_multiple=64)
        assert count_cum.ndim == 0
        # Cumulative head uses raw linear output: untrained model can be negative.
        # Check that it is a finite scalar.
        assert torch.isfinite(count_cum)

        # 2. Cumulative head WITH 4-dir directional context (B5)
        m_b5 = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="cumulative",
            use_integral_context=True,
            output_stride=16,
        )
        field_b5 = m_b5(x)
        assert field_b5.shape == (1, 1, 8, 8)
        assert torch.isfinite(field_b5).all()

        # 3. Local head WITHOUT context (B1 baseline)
        m_b1 = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="local",
            use_integral_context=False,
            output_stride=16,
        )
        field_b1 = m_b1(x)
        assert field_b1.shape == (1, 1, 8, 8)
        # Local head uses softplus: all values must be strictly positive
        assert (field_b1 > 0).all(), "Local head output must be non-negative (softplus)"
        count_b1, _ = m_b1.predict(x, pad_multiple=64)
        assert count_b1.item() > 0

        # 4. Local head WITH directional context (B6)
        m_b6 = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="local",
            use_integral_context=True,
            output_stride=16,
        )
        field_b6 = m_b6(x)
        assert field_b6.shape == (1, 1, 8, 8)
        count_b6, map_b6 = m_b6.predict(x, pad_multiple=64)
        # Local count = sum of all cells; check consistency
        assert abs(count_b6.item() - field_b6.sum().item()) < 1e-4

        # 5. Integrated local head (Section 32 Valid-by-construction baseline)
        m_int = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="integrated_local",
            use_integral_context=False,
            output_stride=16,
        )
        field_int = m_int(x)
        assert field_int.shape == (1, 1, 8, 8)
        # By construction, Delta_xy(C) must be non-negative everywhere
        y_int = discrete_mixed_difference(field_int)
        assert (y_int >= 0).all(), "Integrated local head must be valid by construction (all >= 0)"
        count_int, _ = m_int.predict(x, pad_multiple=64)
        assert count_int.item() > 0

        # 6. Native Stride-8 and Stride-4 routing (verifying no downsampler waste)
        m_s8 = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="cumulative",
            output_stride=8,
        )
        field_s8 = m_s8(x)
        assert field_s8.shape == (1, 1, 16, 16), f"Expected (1, 1, 16, 16), got {field_s8.shape}"

        m_s4 = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="local",
            output_stride=4,
        )
        field_s4 = m_s4(x)
        assert field_s4.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {field_s4.shape}"

        # 7. Hierarchical Tile Composition in predict_tiled
        x_large = torch.randn(1, 3, 256, 512)
        count_tiled, map_tiled = m_cum.predict_tiled(x_large, tile_size=256)
        assert torch.isfinite(count_tiled)
        assert map_tiled.shape == (1, 1, 16, 32)

    def test_axial_integral_context(self):
        from hpc.models.integral_context import (
            AxialIntegralContext,
            _axial_col_prefix,
            _axial_row_prefix,
        )

        module = AxialIntegralContext(channels=16, use_residual=True)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)

        out = module(x)
        assert out.shape == (2, 16, 8, 8)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        with torch.no_grad():
            ones = torch.ones(1, 1, 4, 4)
            r = _axial_row_prefix(ones)
            assert torch.allclose(r, torch.ones_like(r), atol=1e-5)
            v = _axial_col_prefix(ones)
            assert torch.allclose(v, torch.ones_like(v), atol=1e-5)

    def test_micf_lite_axial_context_wiring(self):
        from hpc.models.integral_context import AxialIntegralContext, DirectionalIntegralContext

        m_axial = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="cumulative",
            use_integral_context=True,
            context_type="axial",
            output_stride=16,
        )
        assert isinstance(m_axial.context_module, AxialIntegralContext)

        # Backward compatibility check: use_integral_context=True without context_type builds Directional
        m_directional = MICFLite(
            backbone_name="mobilenetv4_conv_small_050",
            head_type="cumulative",
            use_integral_context=True,
            output_stride=16,
        )
        assert isinstance(m_directional.context_module, DirectionalIntegralContext)

        x = torch.randn(1, 3, 128, 128)
        field = m_axial(x)
        assert field.shape == (1, 1, 8, 8)
        assert torch.isfinite(field).all()

    def test_compose_tiled_cumulative_field(self):
        from hpc.models.micf_lite import compose_tiled_cumulative_field

        torch.manual_seed(42)
        # Validate 50 random (tile grid, tile size) configurations
        for _ in range(50):
            n_tiles_h = torch.randint(1, 5, (1,)).item()
            n_tiles_w = torch.randint(1, 5, (1,)).item()
            tile_h = torch.randint(4, 16, (1,)).item()
            tile_w = torch.randint(4, 16, (1,)).item()

            full_h = n_tiles_h * tile_h
            full_w = n_tiles_w * tile_w

            # Synthetic non-negative discrete cell count map Y
            y_full = torch.rand(full_h, full_w)
            c_expected = torch.cumsum(torch.cumsum(y_full, dim=0), dim=1)

            # Split into tiles and compute independent local cumsum2d per tile
            c_local: list[list[torch.Tensor]] = []
            for i in range(n_tiles_h):
                row_tiles = []
                for j in range(n_tiles_w):
                    y_tile = y_full[i * tile_h:(i + 1) * tile_h, j * tile_w:(j + 1) * tile_w]
                    c_tile = torch.cumsum(torch.cumsum(y_tile, dim=0), dim=1)
                    row_tiles.append(c_tile)
                c_local.append(row_tiles)

            c_composed = compose_tiled_cumulative_field(c_local)
            assert c_composed.shape == c_expected.shape
            assert torch.allclose(c_composed, c_expected, atol=1e-5), (
                f"Mismatch in compose_tiled_cumulative_field: max diff = "
                f"{(c_composed - c_expected).abs().max().item()}"
            )

    def test_points_to_count_map(self):
        from hpc.losses.micf import points_to_count_map

        # Points on a 64x64 image, stride 16 -> 4x4 grid
        # Pixel coordinates: (x, y) with boundary at (x+0.5)/16
        pts = torch.tensor([
            [5.0, 5.0],    # (5+0.5)/16 = 0.34 -> cell (0, 0)
            [10.0, 15.0],  # (10+0.5)/16 = 0.65 -> cell (0, 0)
            [20.0, 35.0],  # x=20->(20.5)/16=1.28->col 1, y=35->(35.5)/16=2.21->row 2 -> cell (2, 1)
            [55.0, 55.0],  # (55.5)/16=3.46 -> cell (3, 3)
        ])
        y = points_to_count_map(pts, out_h=4, out_w=4, stride=16)
        assert y.shape == (4, 4)
        assert y.sum().item() == 4.0
        assert y[0, 0].item() == 2.0
        assert y[2, 1].item() == 1.0
        assert y[3, 3].item() == 1.0

    def test_rectangle_count_recovery(self):
        from hpc.diagnostics.micf_diagnostics import (
            evaluate_rectangle_counts,
            query_rectangle_count,
        )

        # Discrete cell counts
        y = torch.tensor([
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 3.0, 1.0, 0.0],
            [2.0, 0.0, 4.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
        ])
        c = cell_counts_to_cumulative_field(y.unsqueeze(0).unsqueeze(0), orientation="TL")[0, 0]

        # Query sub-rectangle (x in [1, 3], y in [1, 3]) (1-based cell coords)
        # Expected sum: 3 + 1 + 0 + 4 = 8
        rec_count = query_rectangle_count(c, x1=1, y1=1, x2=3, y2=3)
        assert abs(rec_count - 8.0) < 1e-5

        # Query full image [0, 0, 4, 4]
        full_count = query_rectangle_count(c, x1=0, y1=0, x2=4, y2=4)
        assert abs(full_count - y.sum().item()) < 1e-5

        # Multi-scale evaluation
        rect_eval = evaluate_rectangle_counts(c, c, scale_bins=(1/16, 1/4, 1.0))
        assert rect_eval["rectangle_mae_full"] == 0.0
        assert rect_eval["rectangle_mae_large"] == 0.0

    def test_measure_diagnostics(self):
        from hpc.diagnostics.micf_diagnostics import compute_measure_diagnostics

        # Valid cumulative field
        y = torch.ones(1, 1, 4, 4)
        c_valid = cell_counts_to_cumulative_field(y, orientation="TL")
        diag = compute_measure_diagnostics(c_valid)
        assert diag["negative_cell_fraction"] == 0.0
        assert diag["negative_mass_ratio"] == 0.0
        assert diag["violation_magnitude"] == 0.0
        assert diag["n_corner"] == 16.0
        assert diag["n_delta"] == 16.0

    def test_spectral_analysis(self):
        from hpc.diagnostics.micf_diagnostics import compute_spectral_analysis

        # Smooth signal vs noisy signal
        smooth = torch.ones(16, 16)
        noisy = torch.randn(16, 16)

        spec_smooth = compute_spectral_analysis(smooth)
        spec_noisy = compute_spectral_analysis(noisy)

        # Smooth DC signal should have 0 high frequency energy
        assert spec_smooth["high_freq_energy_ratio"] < 1e-4
        # Random noise has much higher high frequency energy
        assert spec_noisy["high_freq_energy_ratio"] > spec_smooth["high_freq_energy_ratio"]


