import json
from pathlib import Path
import tempfile
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
