import os
import argparse
import json
import yaml
import numpy as np
import torch
from PIL import Image

from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics


def run_error_analysis(
    predictions_json: str,
    output_dir: str = "analysis_results",
):
    """Analyze predictions and ground truths and output structured error report."""
    with open(predictions_json, "r") as f:
        data = json.load(f)
        
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truths"])
    lums = np.array(data["luminances"]) if "luminances" in data else None
    
    counting_metrics = evaluate_counting_metrics(preds, gts)
    subgroup_metrics = evaluate_subgroup_diagnostics(preds, gts, lums)
    
    report = {
        "overall_metrics": counting_metrics,
        "subgroup_diagnostics": subgroup_metrics,
        "total_samples": len(preds),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "error_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Error analysis report generated at: {report_path}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save report")
    args = parser.parse_args()
    
    run_error_analysis(args.predictions, args.output_dir)
