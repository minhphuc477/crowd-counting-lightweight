import copy
from pathlib import Path

import pytest
import torch
import yaml

from hpc.models.backbone import MobileNetV4Backbone
from hpc.models.factory import (
    assert_checkpoint_compatible,
    build_model_from_config,
    resolve_pretrained_spec,
    validate_pretrained_normalization,
)
from hpc.models.hpc_lite import HPCLite
from hpc.models.neck import AdditiveFPNNeck
from train_ntpc import build_optimizer


def test_model_forward_and_positivity():
    """HPCLite forward output must have shape (B, 1, H/4, W/4) and be strictly positive."""
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        d_map = model(x)

    assert d_map.shape == (2, 1, 64, 64)
    assert (d_map >= 1e-8).all()
    assert torch.isfinite(d_map).all()


def test_parameter_budget():
    """Dead stride-32 stages must not inflate the deployed parameter budget."""
    model = HPCLite(pretrained=False, neck_width=32)
    n_params = sum(p.numel() for p in model.parameters())
    assert 95_000 <= n_params <= 98_000, f"Parameter count {n_params} drifted outside [95k, 98k]"
    assert model.backbone.truncated_after == "blocks.2"
    assert len(model.backbone.backbone.blocks) == 3


def test_factory_build_model_equivalence():
    """build_model_from_config must build model with matching architecture."""
    cfg = {
        "model": {
            "backbone": "mobilenetv4_conv_small_050",
            "pretrained": False,
            "neck_width": 32,
            "context_dilations": [1, 2, 3],
            "use_p8_context": False,
            "use_repblock": False,
            "eps_d": 1e-8,
        }
    }
    model = build_model_from_config(cfg)
    assert isinstance(model, HPCLite)
    n_params = sum(p.numel() for p in model.parameters())
    assert 95_000 <= n_params <= 98_000


def test_truncated_backbone_is_exactly_equal_to_full_timm_features():
    import timm

    torch.manual_seed(7)
    full = timm.create_model(
        "mobilenetv4_conv_small_050",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3),
    ).eval()
    truncated = MobileNetV4Backbone(
        "mobilenetv4_conv_small_050", pretrained=False
    ).eval()
    incompatible = truncated.backbone.load_state_dict(full.state_dict(), strict=False)
    assert not incompatible.missing_keys
    assert all(key.startswith("blocks.3.") or key.startswith("blocks.4.") for key in incompatible.unexpected_keys)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = full(image)
        actual = truncated(image)
    for expected_level, actual_level in zip(expected, actual):
        torch.testing.assert_close(actual_level, expected_level, rtol=0, atol=0)


def test_truncated_whole_ntpc_forward_and_backward_exact_parity():
    """Removing downstream C32 stages must not change D, count, or retained gradients."""
    import timm

    torch.manual_seed(11)
    truncated = HPCLite(pretrained=False, neck_width=32).eval()
    full = copy.deepcopy(truncated)
    full_feature_extractor = timm.create_model(
        "mobilenetv4_conv_small_050",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3),
    )
    incompatible = full_feature_extractor.load_state_dict(
        truncated.backbone.backbone.state_dict(), strict=False
    )
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith("blocks.3.") or key.startswith("blocks.4.")
        for key in incompatible.missing_keys
    )
    full.backbone.backbone = full_feature_extractor
    full.eval()

    image = torch.randn(2, 3, 64, 64)
    mass_full = full(image)
    mass_truncated = truncated(image)
    torch.testing.assert_close(mass_truncated, mass_full, rtol=0, atol=0)
    torch.testing.assert_close(
        mass_truncated.sum(dim=(-1, -2, -3)),
        mass_full.sum(dim=(-1, -2, -3)),
        rtol=0,
        atol=0,
    )

    mass_full.square().mean().backward()
    mass_truncated.square().mean().backward()
    full_parameters = dict(full.named_parameters())
    for name, parameter in truncated.named_parameters():
        assert parameter.grad is not None, f"Retained parameter {name} has no gradient"
        assert full_parameters[name].grad is not None
        torch.testing.assert_close(parameter.grad, full_parameters[name].grad, rtol=0, atol=0)
    discarded = [
        parameter
        for name, parameter in full.named_parameters()
        if ".blocks.3." in name or ".blocks.4." in name
    ]
    assert discarded
    assert all(parameter.grad is None for parameter in discarded)


def test_truncation_preserves_weights_returned_by_pretrained_factory(monkeypatch):
    """Simulate pretrained creation offline and prove truncation never rewrites retained tensors."""
    import timm

    model_name = "mobilenetv4_conv_small_050"
    original_create_model = timm.create_model
    probe = original_create_model(model_name, pretrained=False, features_only=True)
    loaded = original_create_model(
        model_name,
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3),
    )
    loaded_state_before = {
        key: tensor.detach().clone() for key, tensor in loaded.state_dict().items()
    }
    calls = []

    def fake_create_model(_name, *, pretrained, features_only, out_indices=None):
        calls.append({"pretrained": pretrained, "out_indices": out_indices})
        return probe if len(calls) == 1 else loaded

    monkeypatch.setattr(timm, "create_model", fake_create_model)
    backbone = MobileNetV4Backbone(model_name, pretrained=True)
    assert calls == [
        {"pretrained": False, "out_indices": None},
        {"pretrained": True, "out_indices": (1, 2, 3)},
    ]
    retained_state = backbone.backbone.state_dict()
    assert retained_state
    for key, tensor in retained_state.items():
        torch.testing.assert_close(tensor, loaded_state_before[key], rtol=0, atol=0)


def test_pretrained_contract_and_offline_checkpoint_construction():
    cfg = {
        "model": {
            "backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k",
            "pretrained": True,
        },
        "dataset": {
            "name": "sha",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    }
    spec = validate_pretrained_normalization(cfg)
    assert spec is not None
    assert spec["source"] == "timm/mobilenetv4_conv_small_050.e3000_r224_in1k"
    assert spec["mean"] == (0.5, 0.5, 0.5)

    # Evaluation/checkpoint loading must preserve provenance without downloading weights.
    model = build_model_from_config(cfg, load_pretrained=False)
    assert model.pretrained_requested is True
    assert model.pretrained_loaded is False
    assert model.backbone.pretrained is False

    bad = {
        **cfg,
        "dataset": {
            **cfg["dataset"],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
    }
    with pytest.raises(ValueError, match="normalization mismatch"):
        validate_pretrained_normalization(bad)


def test_published_configs_pin_pretraining_and_normalization():
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "configs/ntpc_one_seed_experiments.yaml").read_text())
    for relative_path in manifest["training_order"]:
        cfg = yaml.safe_load((root / relative_path).read_text())
        assert cfg["model"]["pretrained"] is True
        assert ".e3000_r224_in1k" in cfg["model"]["backbone"]
        assert resolve_pretrained_spec(cfg) is not None
        validate_pretrained_normalization(cfg)
        assert cfg["optimizer"]["backbone_lr_scale"] == pytest.approx(0.1)


def test_ablation_configs_differ_only_in_experiment_and_loss():
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "configs/ntpc_one_seed_experiments.yaml").read_text())
    configs = [yaml.safe_load((root / path).read_text()) for path in manifest["training_order"]]
    controlled_sections = (
        "dataset", "model", "statistics", "augmentation", "sampler",
        "optimizer", "schedule", "training",
    )
    for section in controlled_sections:
        assert all(cfg[section] == configs[0][section] for cfg in configs[1:]), (
            f"Ablation protocol drifted in shared section '{section}'"
        )


def test_discriminative_optimizer_groups_are_disjoint_and_scaled():
    model = HPCLite(pretrained=False, neck_width=32)
    optimizer = build_optimizer(
        model,
        {"name": "AdamW", "lr": 1e-4, "backbone_lr_scale": 0.1, "weight_decay": 1e-4},
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["backbone"]["lr"] == pytest.approx(1e-5)
    assert groups["task"]["lr"] == pytest.approx(1e-4)
    backbone_ids = {id(parameter) for parameter in groups["backbone"]["params"]}
    task_ids = {id(parameter) for parameter in groups["task"]["params"]}
    assert backbone_ids.isdisjoint(task_ids)
    assert backbone_ids | task_ids == {id(parameter) for parameter in model.parameters()}


def test_direct_arbitrary_resolution():
    """HPCLite forward must process arbitrary resolutions directly without zero-padding distortion."""
    model = HPCLite(pretrained=False, neck_width=32).eval()
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        d = model(x)
    assert d.shape == (1, 1, 80, 103)
    assert torch.isfinite(d).all()


def test_arbitrary_direct_and_padded_inference():
    """predict() must handle arbitrary resolutions directly (pad_multiple=None) or with padding."""
    model = HPCLite(pretrained=False, neck_width=32)
    model.eval()

    # Image dimensions not divisible by 32 or 4
    x = torch.randn(1, 3, 317, 411)
    with torch.no_grad():
        count_direct, d_direct = model.predict(x, pad_multiple=None)
        count_padded, d_padded = model.predict(x, pad_multiple=32)

    assert count_direct.ndim == 1 and count_direct.shape[0] == 1
    assert torch.isfinite(count_direct).all()
    assert d_direct.shape == (1, 1, 80, 103)

    assert count_padded.ndim == 1 and count_padded.shape[0] == 1
    assert torch.isfinite(count_padded).all()
    assert d_padded.shape == (1, 1, 80, 103)


@pytest.mark.parametrize("bad_value", [0, -4, 3.5, True])
def test_predict_rejects_invalid_pad_multiple(bad_value):
    model = HPCLite(pretrained=False, neck_width=32).eval()
    with pytest.raises(ValueError, match="positive integer"):
        model.predict(torch.randn(1, 3, 32, 32), pad_multiple=bad_value)


def test_data_driven_head_bias_does_not_modify_pretrained_backbone():
    model = HPCLite(pretrained=False, neck_width=32)
    backbone_before = {
        name: tensor.detach().clone() for name, tensor in model.backbone.state_dict().items()
    }
    bias_before = model.head_out.bias.detach().clone()
    model.init_head_bias_from_data(mean_crop_count=80.0, crop_size=256)
    assert not torch.equal(model.head_out.bias, bias_before)
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(tensor, backbone_before[name], rtol=0, atol=0)


def test_explicit_tiled_inference_shape_count_and_validation():
    model = HPCLite(pretrained=False, neck_width=32).eval()
    image = torch.randn(1, 3, 81, 97)
    count, mass = model.predict_tiled(image, tile_size=64, halo=16)
    assert mass.shape == (1, 1, 21, 25)
    torch.testing.assert_close(count, mass.sum(dim=(-1, -2, -3)))
    assert torch.isfinite(mass).all()
    with pytest.raises(ValueError, match="positive multiple of 16"):
        model.predict_tiled(image, tile_size=63)


def test_checkpoint_compatibility_assertion():
    """assert_checkpoint_compatible must accept matching configs and reject mismatches."""
    cfg = {
        "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
        "dataset": {"name": "sha", "part": "part_A", "coordinate_base": 1},
    }

    # Matching checkpoint
    assert_checkpoint_compatible({"config": cfg}, cfg)

    # Model architecture mismatch
    bad_model_ckpt = {
        "config": {
            "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 64},
            "dataset": {"name": "sha", "part": "part_A", "coordinate_base": 1},
        }
    }
    with pytest.raises(ValueError, match="Model config mismatch"):
        assert_checkpoint_compatible(bad_model_ckpt, cfg)

    # Dataset preprocessing mismatch
    bad_ds_ckpt = {
        "config": {
            "model": {"backbone": "mobilenetv4_conv_small_050", "neck_width": 32},
            "dataset": {"name": "sha", "part": "part_B", "coordinate_base": 1},
        }
    }
    with pytest.raises(ValueError, match="Dataset config mismatch"):
        assert_checkpoint_compatible(bad_ds_ckpt, cfg)


def test_all_cli_modules_import():
    """All tools and CLI modules must import without missing attributes or syntax errors."""
    import tools.eval_localization
    import tools.eval_ntpc_localization_depth
    import tools.architecture_table
    import tools.export_onnx
    import tools.profile_model
    import tools.create_smoke_dataset
    import tools.run_ntpc_one_seed
    import tools.summary_runs
    import tools.visualize_localization


def test_repblock_deploy_parity():
    """RepDWBlock must produce mathematically equivalent output after switch_to_deploy."""
    from hpc.models.blocks import RepDWBlock

    torch.manual_seed(42)
    block = RepDWBlock(channels=32, act=True).eval()
    x = torch.randn(2, 32, 20, 20)

    with torch.no_grad():
        before = block(x)
        block.switch_to_deploy()
        after = block(x)

    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)


def test_arbitrary_resolution_padding_sensitivity():
    """Direct arbitrary resolution prediction vs pad-to-32 must be within small boundary tolerance."""
    model = HPCLite(pretrained=False, neck_width=32).eval()
    x = torch.randn(1, 3, 317, 411)

    with torch.no_grad():
        cnt_direct, _ = model.predict(x, pad_multiple=None)
        cnt_padded, _ = model.predict(x, pad_multiple=32)

    # Relative difference between direct unpadded inference and padded-crop inference should be very small
    rel_diff = abs(float(cnt_direct) - float(cnt_padded)) / max(float(cnt_direct), 1e-4)
    assert rel_diff < 0.05, f"Padding policy sensitivity too high: direct={cnt_direct.item()}, padded={cnt_padded.item()}, rel={rel_diff:.4f}"
