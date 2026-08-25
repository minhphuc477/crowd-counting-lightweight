import os
import argparse
import yaml
import json
import numpy as np

# Configure cache directories on F: disk (avoid C: drive)
_base_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), ".cache"))
os.environ.setdefault("HF_HOME", os.path.join(_base_cache, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_base_cache, "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from hpc.models.hpc_lite import HPCLite
from hpc.data.sha import ShanghaiTechDataset
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.nwpu import NWPUDataset
from hpc.data.transforms import PhotometricTransforms
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from train import build_dataset


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    output_json: str = "eval_results.json",
    eval_robustness: bool = True,
):
    """Run full evaluation protocol with diagnostic subgroups and optional corruption benchmark."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}")
    
    # 1. Load Model
    m_cfg = cfg["model"]
    model = HPCLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050"),
        pretrained=False,
        neck_width=m_cfg.get("neck_width", 32),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        truncate_backbone=m_cfg.get("truncate_backbone", True),
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Running with random/initialized weights.")
        
    model.eval()
    
    # 2. Load Evaluation Dataset
    val_dataset = build_dataset(cfg, is_train=False)
    print(f"Loaded {len(val_dataset)} evaluation samples.")
    
    if len(val_dataset) == 0:
        print("No evaluation samples found.")
        return
        
    predictions = []
    ground_truths = []
    luminances = []
    corrupted_predictions = []
    
    photo_degrader = PhotometricTransforms(
        brightness=0.3,
        contrast=0.3,
        blur_prob=1.0,
        noise_prob=1.0,
    )
    
    print("Running evaluation...")
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
            gt_cnt = float(sample["gt_count"])
            
            # 1. Clean inference
            pred_cnt, _ = model.predict(img, pad_multiple=16)
            pred_val = float(pred_cnt.item())
            
            predictions.append(pred_val)
            ground_truths.append(gt_cnt)
            
            # Compute luminance of the image
            img_np = sample["image"].cpu().numpy()  # (3, H, W)
            lum = float(np.mean(img_np) * 255.0)
            luminances.append(lum)
            
            # 2. Corrupted inference for robustness (if enabled)
            if eval_robustness:
                img_pil = TF.to_pil_image(TF.normalize(
                    sample["image"].cpu(),
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225],
                ).clamp(0, 1))
                deg_pil = photo_degrader(img_pil)
                deg_tensor = TF.normalize(
                    TF.to_tensor(deg_pil),
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ).unsqueeze(0).to(device)
                pred_deg, _ = model.predict(deg_tensor, pad_multiple=16)
                corrupted_predictions.append(float(pred_deg.item()))
                
    counting_metrics = evaluate_counting_metrics(predictions, ground_truths)
    subgroup_metrics = evaluate_subgroup_diagnostics(predictions, ground_truths, luminances)
    
    results = {
        "overall_metrics": counting_metrics,
        "subgroup_diagnostics": subgroup_metrics,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "luminances": luminances,
    }
    
    if eval_robustness and corrupted_predictions:
        corr_mae = float(np.mean(np.abs(np.array(corrupted_predictions) - np.array(ground_truths))))
        delta_mae = corr_mae - counting_metrics["mae"]
        results["robustness_metrics"] = {
            "corrupted_mae": round(corr_mae, 3),
            "delta_mae": round(delta_mae, 3),
        }
        
    print("\n--- Evaluation Results ---")
    print(f"MAE:  {counting_metrics['mae']:.3f}")
    print(f"RMSE: {counting_metrics['rmse']:.3f}")
    print(f"NAE:  {counting_metrics['nae']:.3f}")
    if "robustness_metrics" in results:
        print(f"Corrupted MAE: {results['robustness_metrics']['corrupted_mae']:.3f} (Delta: {results['robustness_metrics']['delta_mae']:+.3f})")
    print(json.dumps(subgroup_metrics, indent=2))
    
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to: {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to output results JSON")
    parser.add_argument("--no_robustness", action="store_true", help="Disable synthetic corruption test")
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.output, eval_robustness=not args.no_robustness)
