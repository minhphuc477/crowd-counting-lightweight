import pytest
import torch

from rmr_v2.losses import LossConfig, compute_losses
from rmr_v2.model import RMRConfig, RMRCount, count_parameters


def test_rmr_v2_model_and_losses():
    cfg = RMRConfig(output_stride=4, feature_width=32, backbone_name="tiny", iterations=2)
    model = RMRCount(cfg, variant="rmr")
    model.eval()

    x = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "regions" in out
    assert out["y"].shape == (1, 1, 16, 16)
    assert (out["y"] >= 0).all()

    target_y = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target_y[0, 0, 4, 4] = 5.0

    losses = compute_losses(out, target_y, variant="rmr", cfg=LossConfig())
    assert "total" in losses
    assert "count" in losses
    assert "flat_dm16" in losses
    assert "cell" in losses
    assert "region_head" in losses
    assert torch.isfinite(losses["total"])


def test_rmr_v2_backward_compatibility():
    import rmr_count
    import rmr_v2
    # Ensure RMRConfig and RMRCount in rmr_v2 and rmr_count match signatures
    cfg1 = rmr_count.model.RMRConfig()
    cfg2 = rmr_v2.RMRConfig()
    assert cfg1.output_stride == cfg2.output_stride
    assert cfg1.feature_width == cfg2.feature_width


def test_rmr_v2_config_hash_sensitivity():
    import yaml
    from rmr_v2.config import compute_v2_config_hash, validate_v2_config, validate_v2_resume_compatibility

    base_cfg = yaml.safe_load(open("configs/rmr_v2/rmr_projected_t2.yaml"))
    validate_v2_config(base_cfg)
    base_hash = compute_v2_config_hash(base_cfg)

    # 1. Changing nb_dispersion must change hash and fail resume validation
    cfg_disp = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg_disp["loss"]["nb_dispersion"] = 5.0
    hash_disp = compute_v2_config_hash(cfg_disp)
    assert base_hash != hash_disp
    with pytest.raises(ValueError, match="Resume config hash mismatch|nb_dispersion"):
        validate_v2_resume_compatibility(base_cfg, cfg_disp, ckpt_hash=base_hash, incoming_hash=hash_disp)

    # 2. Changing detach_region_evidence must change hash and fail resume validation
    cfg_detach = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg_detach["model"]["detach_region_evidence"] = False
    hash_detach = compute_v2_config_hash(cfg_detach)
    assert base_hash != hash_detach
    with pytest.raises(ValueError, match="Resume config hash mismatch|detach_region_evidence"):
        validate_v2_resume_compatibility(base_cfg, cfg_detach, ckpt_hash=base_hash, incoming_hash=hash_detach)

    # 3. Changing update_rule must change hash and fail resume validation
    cfg_rule = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg_rule["model"]["update_rule"] = "latent"
    hash_rule = compute_v2_config_hash(cfg_rule)
    assert base_hash != hash_rule
    with pytest.raises(ValueError, match="Resume config hash mismatch|update_rule"):
        validate_v2_resume_compatibility(base_cfg, cfg_rule, ckpt_hash=base_hash, incoming_hash=hash_rule)

    # 4. Cross-commit without override must fail
    with pytest.raises(ValueError, match="Resume cross-commit mismatch"):
        validate_v2_resume_compatibility(
            base_cfg, base_cfg,
            ckpt_hash=base_hash, incoming_hash=base_hash,
            ckpt_commit="commit_aaa", current_commit="commit_bbb",
            allow_cross_commit=False,
        )

    # 5. Cross-commit with override must pass
    validate_v2_resume_compatibility(
        base_cfg, base_cfg,
        ckpt_hash=base_hash, incoming_hash=base_hash,
        ckpt_commit="commit_aaa", current_commit="commit_bbb",
        allow_cross_commit=True,
    )


def test_rmr_v2_dataset_augmentation_plumbing(tmp_path):
    import json
    from rmr_core.data import CrowdManifestDataset

    manifest_file = tmp_path / "train_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "sample_1", "image": "dummy.jpg", "points": [[10, 10]]}) + "\n")

    cfg = {
        "model": {"output_stride": 4},
        "data": {
            "train_manifest": str(manifest_file),
            "val_manifest": str(manifest_file),
            "crop_size": 256,
            "scale_range": [0.8, 1.2],
            "hflip_prob": 0.8,
            "brightness_jitter": 0.25,
            "contrast_jitter": 0.35,
            "data_root": str(tmp_path),
        },
    }

    d_cfg = cfg.get("data", {})
    train_ds = CrowdManifestDataset(
        d_cfg["train_manifest"],
        train=True,
        output_stride=int(cfg.get("model", {}).get("output_stride", 4)),
        crop_size=int(d_cfg.get("crop_size", 512)),
        scale_range=tuple(d_cfg.get("scale_range", [0.75, 1.25])),
        hflip_prob=float(d_cfg.get("hflip_prob", 0.5)),
        brightness_jitter=float(d_cfg.get("brightness_jitter", 0.0)),
        contrast_jitter=float(d_cfg.get("contrast_jitter", 0.0)),
        data_root=d_cfg.get("data_root"),
    )

    assert train_ds.crop_size == 256
    assert train_ds.scale_range == (0.8, 1.2)
    assert train_ds.hflip_prob == 0.8
    assert train_ds.brightness_jitter == 0.25
    assert train_ds.contrast_jitter == 0.35
    assert train_ds.root == tmp_path

    val_ds = CrowdManifestDataset(
        d_cfg["val_manifest"],
        train=False,
        output_stride=int(cfg.get("model", {}).get("output_stride", 4)),
        data_root=d_cfg.get("data_root"),
    )
    assert val_ds.root == tmp_path


