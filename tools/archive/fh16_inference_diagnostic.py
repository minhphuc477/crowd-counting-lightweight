"""FH-16 inference-only diagnostic for B5b checkpoint.

Purpose
-------
B5b is a global extent-aware model (finite_horizon=None).
At inference time we want to test: what happens if we constrain it to
finite_horizon=16?

Mathematical guarantee (from test_fh_full_horizon_matches_global_extent_aware):
    When grid is exactly K*K (i.e. 16x16 for a 256-crop at stride-16),
    global FH == FH-K=K.  So on training-scale crops the two are identical.
    On full test images the horizon boundary changes.

Loading strategy
----------------
B5b and the FH-K16 model share identical backbone / neck / context weights.
Only the head differs in how it computes the extent condition.
We load the B5b state_dict with strict=False to allow small head mismatches,
then freeze the model and run Full-Direct evaluation on SHA-A test set.

Expected outcomes
-----------------
Direct MAE 209.63 -> ~110:  B5b direct failure is mostly global horizon extrapolation.
Minimal change while B8-K4 stays at ~115: K=4 geometry learning is the key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hpc.data.sha import ShanghaiTechDataset
from hpc.metrics.counting import count_metric_summary
from hpc.models.micf_lite import MICFLite


def _safe_load(path: str, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_config(checkpoint: dict, config_path: str | None) -> dict:
    cfg = checkpoint.get("config")
    if cfg is not None:
        return cfg
    if config_path is None:
        raise ValueError("Checkpoint has no stored config. Provide --config.")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, state_dict: dict, finite_horizon, device) -> MICFLite:
    m_cfg = cfg.get("model", {})
    model = MICFLite(
        backbone_name=m_cfg.get("backbone", "mobilenetv4_conv_small_050.e3000_r224_in1k"),
        pretrained=False,
        neck_width=int(m_cfg.get("neck_width", 32)),
        context_dilations=tuple(m_cfg.get("context_dilations", [1, 2, 3])),
        use_integral_context=bool(m_cfg.get("use_integral_context", False)),
        context_type=str(m_cfg.get("context_type", "directional")),
        head_type=m_cfg.get("head_type", "cumulative"),
        output_stride=int(m_cfg.get("output_stride", 16)),
        eps_d=float(m_cfg.get("eps_d", 1e-8)),
        extent_aware=bool(m_cfg.get("extent_aware", True)),
        finite_horizon=finite_horizon,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
    return model.to(device).eval()


def _build_dataset(cfg, root_override, part_override, split):
    ds_cfg = cfg.get("dataset", {})
    root = root_override or ds_cfg.get("root", "./data/ShanghaiTech")
    part = part_override or ds_cfg.get("part", "part_A")
    return ShanghaiTechDataset(
        root=root, part=part, split=split,
        crop_size=int(ds_cfg.get("crop_size", 256)),
        is_train=False,
        image_mean=ds_cfg.get("image_mean", [0.5, 0.5, 0.5]),
        image_std=ds_cfg.get("image_std", [0.5, 0.5, 0.5]),
        coordinate_base=int(ds_cfg.get("coordinate_base", 0)),
        annotation_bounds_policy="allow",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run B5b checkpoint with finite_horizon=16 at inference (no retrain)."
    )
    p.add_argument("--checkpoint", default="runs/pilot_micf/b5b/best.pt")
    p.add_argument("--config", default=None)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--part", default=None)
    p.add_argument("--split", default="test_data")
    p.add_argument("--pad-multiple", type=int, default=256,
                   help="Must be multiple of stride*FH=16*16=256.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _safe_load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict") or checkpoint

    cfg = _load_config(checkpoint, args.config)
    orig_fh = cfg.get("model", {}).get("finite_horizon", None)

    print("=" * 80)
    print("FH-16 INFERENCE DIAGNOSTIC")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Original FH: {orig_fh}  ->  Inference FH: 16")
    print(f"Epoch stored: {checkpoint.get('epoch')}  |  Best MAE stored: {checkpoint.get('best_mae')}")
    print("=" * 80)

    if args.pad_multiple % 256 != 0:
        print(f"[WARNING] pad_multiple={args.pad_multiple} is not a multiple of 256.")
        print("          FH-16 requires grid divisible by K=16, i.e. pad_multiple must be multiple of stride*K=16*16=256.")

    model_orig = _build_model(cfg, state_dict, finite_horizon=orig_fh, device=device)
    model_fh16 = _build_model(cfg, state_dict, finite_horizon=16, device=device)

    dataset = _build_dataset(cfg, args.dataset_root, args.part, args.split)
    n_eval = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)

    if args.output_dir is None:
        output_dir = checkpoint_path.parent / "eval_fh16"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_orig: list[float] = []
    preds_fh16: list[float] = []
    gts: list[float] = []

    print(f"\nEvaluating {n_eval} images on {device} ...")
    print("-" * 80)

    for idx in range(n_eval):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_count = float(sample["gt_count"].item())

        try:
            count_orig, _ = model_orig.predict(image, pad_multiple=args.pad_multiple)
            pred_orig = float(torch.as_tensor(count_orig).item())
        except Exception as e:
            print(f"[{idx+1:03d}] Original model error: {e}")
            pred_orig = float("nan")

        try:
            count_fh16, _ = model_fh16.predict(image, pad_multiple=args.pad_multiple)
            pred_fh16 = float(torch.as_tensor(count_fh16).item())
        except Exception as e:
            print(f"[{idx+1:03d}] FH-16 model error: {e}")
            pred_fh16 = float("nan")

        ae_orig = abs(pred_orig - gt_count)
        ae_fh16 = abs(pred_fh16 - gt_count)
        preds_orig.append(pred_orig)
        preds_fh16.append(pred_fh16)
        gts.append(gt_count)

        print(
            f"[{idx+1:03d}/{n_eval:03d}] GT={gt_count:7.1f} | "
            f"Orig(FH={orig_fh}) AE={ae_orig:8.2f} | "
            f"FH-16 AE={ae_fh16:8.2f} | delta={ae_orig - ae_fh16:+8.2f}"
        )

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    orig_stats = count_metric_summary(preds_orig, gts, eps=1.0)
    fh16_stats = count_metric_summary(preds_fh16, gts, eps=1.0)

    print(f"{'Metric':<18}{'Original (FH='+str(orig_fh)+')':>22}{'FH-16 Override':>18}{'Delta':>16}")
    print("-" * 74)
    for key in ["mae", "rmse", "nae", "sre", "signed_bias", "median_ae", "p90_ae", "p95_ae", "max_ae"]:
        v_orig = orig_stats[key]
        v_fh16 = fh16_stats[key]
        print(f"{key:<18}{v_orig:>22.4f}{v_fh16:>18.4f}{v_orig-v_fh16:>+16.4f}")

    mae_reduction = orig_stats["mae"] - fh16_stats["mae"]
    print()
    print(f"MAE reduction by FH-16 override: {mae_reduction:+.2f}")
    if mae_reduction > 20:
        print("-> Strong evidence: B5b Direct failure is largely global horizon extrapolation.")
    elif mae_reduction > 5:
        print("-> Moderate evidence: FH-16 horizon helps partially.")
    else:
        print("-> Minimal effect: B5b Direct failure is NOT from global horizon extrapolation.")

    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_mae": checkpoint.get("best_mae"),
        "original_finite_horizon": orig_fh,
        "override_finite_horizon": 16,
        "pad_multiple": args.pad_multiple,
        "num_images": n_eval,
        "original": orig_stats,
        "fh16_override": fh16_stats,
        "mae_reduction_orig_minus_fh16": mae_reduction,
    }

    out_path = output_dir / "fh16_diagnostic_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
