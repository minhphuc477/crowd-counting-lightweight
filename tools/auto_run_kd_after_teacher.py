"""Automated watcher and launcher for Student KD continuation.

Monitors TeacherLite training until completion, checks best checkpoint,
and immediately triggers Student KD training (Phase Q).
"""
import csv
import os
import subprocess
import sys
import time


def check_teacher_status(val_csv_path: str, target_epochs: int = 1000):
    if not os.path.exists(val_csv_path):
        return 0, float("inf"), False

    last_epoch = 0
    best_mae = float("inf")
    with open(val_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["epoch"])
                mae = float(row["mae_map"])
                last_epoch = max(last_epoch, ep)
                best_mae = min(best_mae, mae)
            except Exception:
                continue

    is_done = (last_epoch >= target_epochs)
    return last_epoch, best_mae, is_done


def main():
    val_csv = "./runs/sha_teacher_lite/val.csv"
    teacher_ckpt = "./runs/sha_teacher_lite/best.pt"
    kd_config = "configs/sha_kd_quick.yaml"

    print("==========================================================")
    print(" AUTO-LAUNCHER: Waiting for TeacherLite training to finish")
    print("==========================================================")
    print(f"Monitoring: {val_csv}")
    print(f"Target KD config: {kd_config}")
    print("----------------------------------------------------------")

    while True:
        last_ep, best_mae, is_done = check_teacher_status(val_csv, target_epochs=1000)
        print(f"[Watcher] Teacher Epoch: {last_ep:04d}/1000 | Best Map MAE: {best_mae:.2f}", flush=True)

        if is_done:
            print("\n[Watcher] TeacherLite training complete!")
            break

        time.sleep(30)

    # Verify best checkpoint
    if not os.path.exists(teacher_ckpt):
        print(f"ERROR: Expected teacher checkpoint {teacher_ckpt} not found!")
        sys.exit(1)

    print(f"[Watcher] Verified Teacher checkpoint: {teacher_ckpt} (Best Map MAE = {best_mae:.2f})")
    print("\n==========================================================")
    print(" LAUNCHING STUDENT KD CONTINUATION (Phase Q)")
    print("==========================================================")

    cmd = [
        sys.executable,
        "-u",
        "train_kd.py",
        "--config",
        kd_config,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
