import argparse
import torch
import torch.nn as nn
from hpc.teachers.teacher_lite import TeacherLite
from hpc.losses.teacher_criterion import TeacherCriterion
from hpc.losses.kd import MultiLevelKDLoss
from hpc.models.hpc_lite import HPCLiteSR48, HPCLite
from hpc.losses.criterion import HPCLossCriterion


def mb(x):
    return x / (1024 ** 2)


def profile_teacher(teacher, criterion, batch, optimizer, use_bf16=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = teacher(batch["image"])
        loss, _ = criterion(out, batch)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    return {
        "allocated_MB": mb(torch.cuda.max_memory_allocated()),
        "reserved_MB": mb(torch.cuda.max_memory_reserved()),
    }


def profile_kd(student, teacher, gt_criterion, kd_criterion, batch, optimizer, use_bf16=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            teacher_out = teacher(batch["image"])

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        d_s, aux_s = student(batch["image"], return_aux=True)
        student_out = {"density": d_s, "p4": aux_s["p4"], "p8": aux_s["p8"], "p16": aux_s["p16"]}

        gt_loss, _ = gt_criterion(
            d_map=student_out["density"],
            gt_block_counts=batch["gt_blocks"],
            gt_counts=batch["gt_count"],
            d_degraded=None,
            progress=0.5,
        )
        kd_loss, _ = kd_criterion(
            student_out,
            teacher_out,
            batch["gt_count"],
            progress=0.5,
        )
        (gt_loss + kd_loss).backward()
    optimizer.step()
    torch.cuda.synchronize()

    return {
        "allocated_MB": mb(torch.cuda.max_memory_allocated()),
        "reserved_MB": mb(torch.cuda.max_memory_reserved()),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available, skipping VRAM profile.")
        return

    device = torch.device("cuda")
    print(f"Profiling VRAM on GPU: {torch.cuda.get_device_name(0)}")

    for batch_size in [2, 4, 8, 16]:
        print(f"\n--- Batch Size = {batch_size} ---")
        img = torch.randn(batch_size, 3, 448, 448, device=device)
        gt_count = torch.full((batch_size,), 50.0, device=device)
        gt_z_alloc = torch.zeros(batch_size, 1, 112, 112, device=device)
        gt_blocks = {
            16: torch.zeros(batch_size, 28, 28, device=device),
            32: torch.zeros(batch_size, 14, 14, device=device),
            64: torch.zeros(batch_size, 7, 7, device=device),
        }
        batch = {
            "image": img,
            "gt_count": gt_count,
            "gt_z_alloc": gt_z_alloc,
            "gt_blocks": gt_blocks,
        }

        # Profile Teacher
        teacher = TeacherLite(width=96, pretrained=False).to(device)
        teacher_crit = TeacherCriterion().to(device)
        opt_t = torch.optim.AdamW(teacher.parameters(), lr=1e-4)
        t_res = profile_teacher(teacher, teacher_crit, batch, opt_t)
        print(f"Teacher Training : Allocated = {t_res['allocated_MB']:.1f} MB | Reserved = {t_res['reserved_MB']:.1f} MB")
        del teacher, teacher_crit, opt_t
        torch.cuda.empty_cache()

        # Profile KD Student
        student = HPCLiteSR48(pretrained=False, neck_width=48).to(device)
        teacher = TeacherLite(width=96, pretrained=False).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        gt_crit = HPCLossCriterion(block_sizes=[16, 32, 64]).to(device)
        kd_crit = MultiLevelKDLoss(
            student_channels={"p4": 48, "p8": 48, "p16": 48},
            teacher_channels={"p4": 96, "p8": 96, "p16": 96},
        ).to(device)
        opt_s = torch.optim.AdamW(list(student.parameters()) + list(kd_crit.parameters()), lr=1e-4)
        kd_res = profile_kd(student, teacher, gt_crit, kd_crit, batch, opt_s)
        print(f"Student KD Train : Allocated = {kd_res['allocated_MB']:.1f} MB | Reserved = {kd_res['reserved_MB']:.1f} MB")
        del student, teacher, gt_crit, kd_crit, opt_s
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
