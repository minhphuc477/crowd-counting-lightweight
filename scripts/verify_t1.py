from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.metrics import density_stratified_mae, summarize_predictions
from rmr_v3.eval import load_model_from_ckpt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path("runs/sha_a/rmr_v7_ablation_no_hurdle/best_val_mae.pt")
    if not ckpt_path.exists():
        print("Checkpoint does not exist:", ckpt_path)
        return

    model, _, cfg, _ = load_model_from_ckpt(ckpt_path, device, use_ema=True)
    model.eval()

    dataset = CrowdManifestDataset("data/sha_a_test.jsonl", train=False, output_stride=4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval)

    for t in [0, 1, 2, 4]:
        model.cfg.iterations = t
        rows = []
        with torch.no_grad():
            for batch in loader:
                sample = batch[0]
                x = sample["image"].unsqueeze(0).to(device)
                out = model(x)
                pred = float(out["y"].sum().item())
                gt = float(sample["target_y"].sum().item())
                rows.append({"pred": pred, "gt": gt})
        sum_res = summarize_predictions(rows)
        strat = density_stratified_mae(rows)
        print(
            f"T={t}: MAE={sum_res['MAE']:.2f}, RMSE={sum_res['RMSE']:.2f}, "
            f"Sparse={strat.get('sparse_le100', 0):.2f}, Mod={strat.get('mid_101_500', 0):.2f}, "
            f"Dense={strat.get('dense_gt500', 0):.2f}"
        )


if __name__ == "__main__":
    main()
