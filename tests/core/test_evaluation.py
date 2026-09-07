import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rmr_core.evaluation import evaluate_dataset, predict_tiled, save_evaluation_artifacts


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)

    def forward(self, x, **kwargs):
        # Return constant positive mass of 100 per sample
        b = x.shape[0]
        y = torch.full((b, 1, 16, 16), 100.0 / 256.0, device=x.device)
        return {"y": y}


def test_evaluate_dataset_and_artifacts():
    device = torch.device("cpu")
    model = DummyModel()

    sample_batch = [
        {
            "image": torch.zeros((3, 64, 64)),
            "target_y": torch.zeros((1, 16, 16)),
            "points": torch.zeros((80, 2)),
            "id": "img_001",
            "height": 64,
            "width": 64,
        },
        {
            "image": torch.zeros((3, 64, 64)),
            "target_y": torch.zeros((1, 16, 16)),
            "points": torch.zeros((120, 2)),
            "id": "img_002",
            "height": 64,
            "width": 64,
        },
    ]
    # loader yields 1 batch containing 2 samples
    loader = [sample_batch]

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=4,
        run_tiling=False,
    )

    assert len(rows) == 2
    assert rows[0]["id"] == "img_001"
    assert rows[1]["id"] == "img_002"
    assert abs(rows[0]["pred"] - 100.0) < 1e-4

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sum_p, pred_p = save_evaluation_artifacts(tmp_path, rows, summary)
        assert sum_p.exists()
        assert pred_p.exists()

        content = sum_p.read_text(encoding="utf-8")
        data = json.loads(content)
        assert "MAE" in data
        assert "mae_ci95" in data


class LocalConvModel(nn.Module):
    """Local convolutional operator where direct forward algebraically equals tiled prediction."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1, stride=4, bias=True)
        nn.init.constant_(self.conv.weight, 0.25)
        nn.init.constant_(self.conv.bias, 0.05)

    def forward(self, x, **kwargs):
        return {"y": torch.relu(self.conv(x))}


def test_tiled_prediction_invariant():
    """Verify the mathematical invariant: Direct Forward == Tiled Forward for local models."""
    torch.manual_seed(42)
    model = LocalConvModel().eval()
    image = torch.rand(3, 128, 128)

    with torch.no_grad():
        direct_out = model(image.unsqueeze(0))["y"][0]
        tiled_out = predict_tiled(
            model=model,
            image=image,
            output_stride=4,
            tile_size=64,
            halo=16,
        )

    assert direct_out.shape == tiled_out.shape
    assert torch.allclose(direct_out, tiled_out, atol=1e-6)


def test_evaluate_dataset_with_tiling():
    """Verify evaluate_dataset computes tiling discrepancies and metrics when run_tiling=True."""
    device = torch.device("cpu")
    model = LocalConvModel()

    sample_batch = [
        {
            "image": torch.rand((3, 64, 64)),
            "target_y": torch.ones((1, 16, 16)) * 0.1,
            "points": torch.zeros((10, 2)),
            "id": "img_tile_001",
            "height": 64,
            "width": 64,
        }
    ]
    loader = [sample_batch]

    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=4,
        run_tiling=True,
        tile_size=32,
        practical_halo=8,
    )

    assert len(rows) == 1
    assert "pred_tiled_h0" in rows[0]
    assert "pred_tiled_practical" in rows[0]
    assert "direct_tiled_discrepancy_mean" in summary
    assert "direct_tiled_h0_discrepancy_mean" in summary
    assert "MAE" in summary


def test_gt_consistency_invariant():
    """Verify that enforce_gt_consistency raises ValueError on discrepancy between raster sum and raw points."""
    device = torch.device("cpu")
    model = LocalConvModel()

    sample_batch = [
        {
            "image": torch.rand((3, 64, 64)),
            "target_y": torch.ones((1, 16, 16)),  # sum = 256
            "points": torch.tensor([[10.0, 10.0], [20.0, 20.0]]),  # 2 points
            "id": "img_mismatch",
            "height": 64,
            "width": 64,
        }
    ]
    loader = [sample_batch]

    with pytest.raises(ValueError, match="GT consistency invariant violated"):
        evaluate_dataset(
            model=model,
            loader=loader,
            device=device,
            enforce_gt_consistency=True,
        )

