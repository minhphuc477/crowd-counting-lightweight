"""Central construction of official evaluation datasets and splits."""

from __future__ import annotations

from .nwpu import NWPUDataset, resolve_nwpu_split_file
from .qnrf import UCFQNRFDataset
from .sha import ShanghaiTechDataset


def build_evaluation_dataset(cfg: dict, split: str | None = None):
    """Build an evaluation dataset and return ``(dataset, resolved_split)``.

    Keeping this policy in one place prevents counting, localization, and
    visualization tools from silently evaluating different preprocessing/splits.
    """
    ds_cfg = cfg["dataset"]
    name = str(ds_cfg.get("name", "sha")).lower().replace("-", "_")
    common = {
        "crop_size": int(ds_cfg.get("crop_size", 256)),
        "is_train": False,
        "image_mean": ds_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": ds_cfg.get("image_std", [0.229, 0.224, 0.225]),
    }
    if "coordinate_base" in ds_cfg:
        common["coordinate_base"] = int(ds_cfg["coordinate_base"])

    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = ds_cfg.get("part", "part_B" if name.endswith("_b") else "part_A")
        resolved_split = split or "test_data"
        dataset = ShanghaiTechDataset(
            root=ds_cfg["root"], part=part, split=resolved_split, **common
        )
    elif name in {"qnrf", "ucf_qnrf"}:
        resolved_split = split or "Test"
        dataset = UCFQNRFDataset(root=ds_cfg["root"], split=resolved_split, **common)
    elif name == "nwpu":
        resolved_split = split or "val"
        dataset = NWPUDataset(
            root=ds_cfg["root"],
            split=resolved_split,
            split_file=resolve_nwpu_split_file(ds_cfg, resolved_split),
            **common,
        )
    else:
        raise ValueError(f"Unsupported dataset '{name}'")
    return dataset, resolved_split
