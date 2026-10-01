import subprocess
import sys
from pathlib import Path

v25_models = [
    "rmr_v25_ablation_no_bb",
    "rmr_v25_ablation_no_perspective",
    "rmr_v25_ablation_no_curvature",
    "rmr_v25_ablation_no_morozov",
    "rmr_v25_control_no_solver",
]

py_exe = sys.executable

for model in v25_models:
    ckpt = Path("runs/sha_a") / model / "best_val_mae.pt"
    if not ckpt.exists():
        print(f"Skipping {model}: {ckpt} does not exist.")
        continue
    print(f"\n=======================================================")
    print(f"Running TTA Evaluation on {model}...")
    print(f"=======================================================")
    cmd = [
        py_exe,
        "-m",
        "rmr_v3.eval",
        "--checkpoint",
        str(ckpt),
        "--tta",
        "--no-tiling",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error evaluating {model}:\n{res.stderr}")
    else:
        # Print the relevant evaluation output
        lines = res.stdout.strip().split("\n")
        eval_lines = [l for l in lines if any(k in l for k in ["MAE:", "Iterates", "Sparse", "Moderate", "Dense"])]
        for el in eval_lines:
            print("  " + el)
print("\nCompleted all TTA evaluations.")
