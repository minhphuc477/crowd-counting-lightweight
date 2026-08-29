import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.teachers.teacher_lite import TeacherLite
from hpc.losses.teacher_criterion import TeacherCriterion
from hpc.losses.kd import MultiLevelKDLoss, normalized_mass
from hpc.models.hpc_lite import HPCLiteSR48, HPCLite
from hpc.losses.criterion import HPCLossCriterion


def test_1_teacher_shapes():
    """Test 1 — Teacher output shapes."""
    teacher = TeacherLite(width=96, pretrained=False)
    teacher.eval()
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        out = teacher(x)
    
    assert out["density"].shape == (2, 1, 112, 112), f"Got {out['density'].shape}"
    assert out["p4"].shape == (2, 96, 112, 112), f"Got {out['p4'].shape}"
    assert out["p8"].shape == (2, 96, 56, 56), f"Got {out['p8'].shape}"
    assert out["p16"].shape == (2, 96, 28, 28), f"Got {out['p16'].shape}"
    assert out["p32"].shape == (2, 96, 14, 14), f"Got {out['p32'].shape}"
    assert out["count_map"].shape == (2,), f"Got {out['count_map'].shape}"
    assert out["count_reg"].shape == (2,), f"Got {out['count_reg'].shape}"


def test_2_teacher_positivity():
    """Test 2 — Teacher positivity and finite values."""
    teacher = TeacherLite(width=96, pretrained=False)
    teacher.eval()
    x = torch.randn(2, 3, 448, 448)
    with torch.no_grad():
        out = teacher(x)

    assert torch.isfinite(out["density"]).all()
    assert (out["density"] > 0).all()
    assert torch.isfinite(out["count_reg"]).all()
    assert (out["count_reg"] >= 0).all()
    assert torch.isfinite(out["count_map"]).all()
    assert (out["count_map"] > 0).all()


def test_3_kd_no_teacher_gradient():
    """Test 3 — Teacher must have zero gradients during KD backward."""
    student = HPCLiteSR48(pretrained=False, neck_width=48)
    teacher = TeacherLite(width=96, pretrained=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    kd_loss = MultiLevelKDLoss(
        student_channels={"p4": 48, "p8": 48, "p16": 48},
        teacher_channels={"p4": 96, "p8": 96, "p16": 96},
        kd_dim=64,
    )

    x = torch.randn(2, 3, 448, 448)
    gt_cnt = torch.tensor([50.0, 100.0])

    with torch.no_grad():
        t_out = teacher(x)

    d_s, aux_s = student(x, return_aux=True)
    s_out = {"density": d_s, "p4": aux_s["p4"], "p8": aux_s["p8"], "p16": aux_s["p16"]}

    loss, _ = kd_loss(s_out, t_out, gt_cnt, progress=0.5)
    loss.backward()

    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter {name} received gradient!"


def test_4_kd_projectors_receive_gradient():
    """Test 4 — KD projectors do receive gradients."""
    student = HPCLiteSR48(pretrained=False, neck_width=48)
    teacher = TeacherLite(width=96, pretrained=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    kd_loss = MultiLevelKDLoss(
        student_channels={"p4": 48, "p8": 48, "p16": 48},
        teacher_channels={"p4": 96, "p8": 96, "p16": 96},
        kd_dim=64,
    )

    x = torch.randn(2, 3, 448, 448)
    gt_cnt = torch.tensor([50.0, 100.0])

    with torch.no_grad():
        t_out = teacher(x)

    d_s, aux_s = student(x, return_aux=True)
    s_out = {"density": d_s, "p4": aux_s["p4"], "p8": aux_s["p8"], "p16": aux_s["p16"]}

    loss, _ = kd_loss(s_out, t_out, gt_cnt, progress=0.5)
    loss.backward()

    has_student_proj_grad = any(p.grad is not None for p in kd_loss.student_proj.parameters())
    has_teacher_proj_grad = any(p.grad is not None for p in kd_loss.teacher_proj.parameters())

    assert has_student_proj_grad, "student_proj must receive gradients"
    assert has_teacher_proj_grad, "teacher_proj must receive gradients"


def test_5_deployment_param_invariance():
    """Test 5 — Deployed student parameters must not change with KD."""
    student = HPCLiteSR48(pretrained=False, neck_width=48)
    params_before = sum(p.numel() for p in student.parameters())

    # Build KD modules
    teacher = TeacherLite(width=96, pretrained=False)
    kd_loss = MultiLevelKDLoss(
        student_channels={"p4": 48, "p8": 48, "p16": 48},
        teacher_channels={"p4": 96, "p8": 96, "p16": 96},
        kd_dim=64,
    )

    # Deployed student parameter count after setup
    params_after = sum(p.numel() for p in student.parameters())
    assert params_before == params_after == 173909, f"Expected 173909 params, got {params_after}"


def test_6_zero_kd_reproduces_gt_loss():
    """Test 6 — Zero KD lambdas reproduce pure GT training loss."""
    gt_criterion = HPCLossCriterion(
        block_sizes=[16, 32, 64],
        lambda_count=1.0,
        lambda_hnb=0.35,
        lambda_alloc=0.15,
    )
    kd_criterion = MultiLevelKDLoss(
        student_channels={"p4": 48, "p8": 48, "p16": 48},
        teacher_channels={"p4": 96, "p8": 96, "p16": 96},
        lambda_feat=0.0,
        lambda_energy=0.0,
        lambda_relation=0.0,
        lambda_map=0.0,
        lambda_count=0.0,
    )

    student = HPCLiteSR48(pretrained=False, neck_width=48)
    teacher = TeacherLite(width=96, pretrained=False)

    x = torch.randn(2, 3, 448, 448)
    gt_cnt = torch.tensor([50.0, 100.0])
    gt_blocks = {
        16: torch.zeros(2, 28, 28),
        32: torch.zeros(2, 14, 14),
        64: torch.zeros(2, 7, 7),
    }

    d_s, aux_s = student(x, return_aux=True)
    s_out = {"density": d_s, "p4": aux_s["p4"], "p8": aux_s["p8"], "p16": aux_s["p16"]}
    t_out = teacher(x)

    gt_loss, _ = gt_criterion(d_s, gt_blocks, gt_cnt)
    kd_loss_val, _ = kd_criterion(s_out, t_out, gt_cnt, progress=0.5)

    assert torch.allclose(gt_loss + kd_loss_val, gt_loss, atol=1e-6)


def test_7_map_kd_scale_invariance():
    """Test 7 — Map KD scale invariance under scalar multiplication."""
    kd_loss = MultiLevelKDLoss(
        student_channels={"p4": 48, "p8": 48, "p16": 48},
        teacher_channels={"p4": 96, "p8": 96, "p16": 96},
    )

    ds = torch.rand(2, 1, 112, 112) + 0.1
    dt = torch.rand(2, 1, 112, 112) + 0.1

    l1 = kd_loss.map_loss(ds, dt)
    l2 = kd_loss.map_loss(ds, 7.0 * dt)

    assert torch.allclose(l1, l2, atol=1e-5, rtol=1e-4)


def test_8_teacher_criterion_backward():
    """Test 8 — Teacher Criterion finite loss and backward."""
    teacher = TeacherLite(width=96, pretrained=False)
    criterion = TeacherCriterion()

    x = torch.randn(2, 3, 448, 448)
    out = teacher(x)

    batch = {
        "gt_count": torch.tensor([25.0, 50.0]),
        "gt_z_alloc": torch.zeros(2, 1, 112, 112),
        "gt_blocks": {
            16: torch.zeros(2, 28, 28),
            32: torch.zeros(2, 14, 14),
            64: torch.zeros(2, 7, 7),
        },
    }

    loss, details = criterion(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in teacher.parameters())
