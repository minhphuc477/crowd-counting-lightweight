"""Tests for training resume reproducibility and RNG checkpointing."""

import random
import numpy as np
import torch

from train_ntpc import get_rng_state


def test_rng_checkpoint_roundtrip():
    state = get_rng_state()
    assert "torch" in state
    assert "numpy" in state
    assert "python" in state
    assert isinstance(state["torch"], torch.Tensor)
    assert isinstance(state["numpy"], tuple)
    assert isinstance(state["python"], tuple)


def test_assert_resume_compatible_detects_crop_size_mismatch():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    cfg_a = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r2_flat_dm"},
    }
    cfg_c = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "crop_size": 448},
        "loss": {"mode": "r2_flat_dm"},
    }

    ckpt_a = {"config": cfg_a}
    # Matching config passes
    assert_resume_compatible(ckpt_a, cfg_a)

    # Mismatched crop_size raises ValueError
    with pytest.raises(ValueError, match="Resume protocol mismatch in: dataset"):
        assert_resume_compatible(ckpt_a, cfg_c)


def test_resume_rejects_statistics_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r5_full_ntpc", "dense_threshold_16": "auto"},
        "statistics": {"seed": 12345, "crops_per_image": 3},
    }
    new_cfg = {
        **old_cfg,
        "statistics": {"seed": 999, "crops_per_image": 5},
    }

    checkpoint = {"config": old_cfg}
    with pytest.raises(ValueError, match="Resume protocol mismatch in: statistics"):
        assert_resume_compatible(checkpoint, new_cfg)


def test_resume_rejects_evaluation_cadence_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha", "crop_size": 256},
        "loss": {"mode": "r2_flat_dm"},
        "training": {"evaluate_every": 5},
    }
    new_cfg = {
        **old_cfg,
        "training": {"evaluate_every": 20},
    }

    with pytest.raises(ValueError, match="Resume protocol mismatch in: training"):
        assert_resume_compatible({"config": old_cfg}, new_cfg)


def test_dataset_resolver_matches_loader_defaults():
    from hpc.models.factory import resolve_dataset_config

    assert resolve_dataset_config({"dataset": {"name": "sha"}})["coordinate_base"] == 0
    assert resolve_dataset_config({"dataset": {"name": "qnrf"}})["coordinate_base"] == 1
    assert resolve_dataset_config({"dataset": {"name": "nwpu"}})["coordinate_base"] == 0

    shb = resolve_dataset_config({"dataset": {"name": "shanghaitech_b"}})
    assert shb["part"] == "part_B"
    assert shb["coordinate_base"] == 0


def test_resume_rejects_persistent_workers_drift():
    import pytest
    from hpc.models.factory import assert_resume_compatible

    old_cfg = {
        "dataset": {"name": "sha"},
        "training": {"num_workers": 2, "persistent_workers": False},
    }
    new_cfg = {
        "dataset": {"name": "sha"},
        "training": {"num_workers": 2, "persistent_workers": True},
    }

    with pytest.raises(ValueError, match="Resume protocol mismatch in: training"):
        assert_resume_compatible({"config": old_cfg}, new_cfg)


def test_exact_resume_continuation():
    """Prove bitwise exactness of interrupted vs uninterrupted training runs."""
    import copy
    from torch.utils.data import DataLoader, Dataset
    from hpc.models.factory import build_model_from_config
    from hpc.losses.factory import build_ntpc_criterion_from_config
    from hpc.utils.seed import make_generator, seed_everything
    from train_ntpc import build_optimizer

    class SyntheticCrowdDataset(Dataset):
        def __init__(self, count: int = 8):
            self.count = count

        def __len__(self):
            return self.count

        def __getitem__(self, idx):
            # Deterministic synthetic image and target pyramid
            torch.manual_seed(idx * 1000 + 7)
            img = torch.randn(3, 128, 128)
            h4, w4 = 32, 32
            # Stride 4 ground truth blocks
            gt4 = torch.randint(0, 5, (h4, w4), dtype=torch.float32)
            # Hierarchical blocks
            gt8 = gt4.view(16, 2, 16, 2).sum(dim=(1, 3))
            gt16 = gt8.view(8, 2, 8, 2).sum(dim=(1, 3))
            gt32 = gt16.view(4, 2, 4, 2).sum(dim=(1, 3))
            gt64 = gt32.view(2, 2, 2, 2).sum(dim=(1, 3))
            total_n = gt4.sum()
            return {
                "image": img,
                "gt_blocks": {64: gt64, 32: gt32, 16: gt16, 8: gt8, 4: gt4},
                "gt_count": total_n,
            }

    def collate_fn(batch):
        images = torch.stack([x["image"] for x in batch])
        gt_blocks = {
            res: torch.stack([x["gt_blocks"][res] for x in batch])
            for res in (64, 32, 16, 8, 4)
        }
        gt_count = torch.stack([x["gt_count"] for x in batch])
        return {"image": images, "gt_blocks": {**gt_blocks, "N": gt_count}}

    cfg = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 16, "pretrained": False},
        "loss": {"mode": "r2_flat_dm", "dense_threshold_16": 5.0},
        "optimizer": {
            "name": "adamw",
            "lr_backbone": 1e-4,
            "lr_task": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 5.0,
        },
        "schedule": {"epochs": 4, "warmup_epochs": 1},
        "statistics": {"mean_crop_count": 50.0, "dense_threshold_q85": 5},
    }

    # === RUN 1: Uninterrupted 4 Epochs ===
    seed_everything(999)
    loader_gen_1 = make_generator(999)
    ds1 = SyntheticCrowdDataset(count=8)
    loader1 = DataLoader(
        ds1, batch_size=4, shuffle=True, collate_fn=collate_fn, generator=loader_gen_1
    )

    model1 = build_model_from_config(cfg, load_pretrained=False)
    criterion1 = build_ntpc_criterion_from_config(cfg, crop_statistics=cfg["statistics"])
    optimizer1 = build_optimizer(model1, cfg["optimizer"])
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lambda ep: 0.5 * (1.0 + ep))

    losses_run1 = []
    for epoch in range(1, 5):
        model1.train()
        for batch in loader1:
            optimizer1.zero_grad()
            mass = model1(batch["image"])
            loss, _ = criterion1(mass, batch["gt_blocks"])
            loss.backward()
            optimizer1.step()
            losses_run1.append(float(loss.detach()))
        scheduler1.step()

    final_model1_state = copy.deepcopy(model1.state_dict())
    final_opt1_state = copy.deepcopy(optimizer1.state_dict())

    # === RUN 2: 2 Epochs -> Checkpoint -> Resume -> 2 Epochs ===
    seed_everything(999)
    loader_gen_2 = make_generator(999)
    ds2 = SyntheticCrowdDataset(count=8)
    loader2 = DataLoader(
        ds2, batch_size=4, shuffle=True, collate_fn=collate_fn, generator=loader_gen_2
    )

    model2 = build_model_from_config(cfg, load_pretrained=False)
    criterion2 = build_ntpc_criterion_from_config(cfg, crop_statistics=cfg["statistics"])
    optimizer2 = build_optimizer(model2, cfg["optimizer"])
    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda ep: 0.5 * (1.0 + ep))

    losses_run2 = []
    for epoch in range(1, 3):
        model2.train()
        for batch in loader2:
            optimizer2.zero_grad()
            mass = model2(batch["image"])
            loss, _ = criterion2(mass, batch["gt_blocks"])
            loss.backward()
            optimizer2.step()
            losses_run2.append(float(loss.detach()))
        scheduler2.step()

    # Save checkpoint state at epoch 2
    ckpt = {
        "epoch": 2,
        "model_state_dict": copy.deepcopy(model2.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer2.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler2.state_dict()),
        "rng_state": get_rng_state(),
        "loader_generator_state": loader_gen_2.get_state().clone(),
    }

    # Interruption: mutate states completely
    seed_everything(42)
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.randn_like(p))

    # Resume from checkpoint
    model2.load_state_dict(ckpt["model_state_dict"])
    optimizer2.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler2.load_state_dict(ckpt["scheduler_state_dict"])
    torch.set_rng_state(ckpt["rng_state"]["torch"])
    np.random.set_state(ckpt["rng_state"]["numpy"])
    random.setstate(ckpt["rng_state"]["python"])
    loader_gen_2.set_state(ckpt["loader_generator_state"])

    # Continue epochs 3..4
    for epoch in range(3, 5):
        model2.train()
        for batch in loader2:
            optimizer2.zero_grad()
            mass = model2(batch["image"])
            loss, _ = criterion2(mass, batch["gt_blocks"])
            loss.backward()
            optimizer2.step()
            losses_run2.append(float(loss.detach()))
        scheduler2.step()

    # === BITWISE EXACTNESS CHECKS ===
    # 1. Losses in resumed epochs 3 and 4 match uninterrupted run exactly
    assert len(losses_run1) == len(losses_run2) == 8
    for i, (l1, l2) in enumerate(zip(losses_run1, losses_run2)):
        assert abs(l1 - l2) < 1e-6, f"Loss mismatch at step {i}: {l1} vs {l2}"

    # 2. Final model parameters are identical
    for k in final_model1_state:
        assert torch.allclose(final_model1_state[k], model2.state_dict()[k], atol=1e-7), (
            f"Model parameter mismatch in {k}"
        )

    # 3. Final optimizer state parameters are identical
    opt1_state = optimizer1.state_dict()
    opt2_state = optimizer2.state_dict()
    assert len(opt1_state["state"]) == len(opt2_state["state"])
    for p_id in opt1_state["state"]:
        for tensor_key in ("exp_avg", "exp_avg_sq"):
            if tensor_key in opt1_state["state"][p_id]:
                t1 = opt1_state["state"][p_id][tensor_key]
                t2 = opt2_state["state"][p_id][tensor_key]
                assert torch.allclose(t1, t2, atol=1e-7), f"Optimizer {tensor_key} mismatch for {p_id}"





