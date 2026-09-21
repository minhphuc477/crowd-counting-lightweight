from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass
class RMRv3LossConfig:
    # Primary loss weights
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.50
    lambda_region_nb: float = 0.20
    lambda_hurdle: float = 0.10
    lambda_trunc_nb: float = 0.20

    # Loss mode selection
    allocation_loss_type: str = "flat_dm16"  # "flat_dm16" | "bayesian" | "ot_sinkhorn"
    bayesian_sigma: float = 8.0
    bayesian_background_ratio: float = 0.10
    ot_reg: float = 10.0
    ot_num_iters: int = 20

    # Count loss configuration
    count_loss_mode: str = "nb"  # "nb" | "log1p" | "l1"
    count_nb_dispersion: float = 50.0

    # Allocation loss configuration
    output_stride: int = 4
    kappa_flat16: float = 20.0
    normalize_flat_dm16: bool = True
    dm_strict: bool = False
    dm_target: str = "y"  # "y" | "y0" | "dual"

    # Multi-scale / Hierarchical DM loss
    use_multiscale_dm: bool = False
    use_hierarchical_dm: bool = False
    dm_block_sizes_px: tuple[int, ...] = (16, 32, 64)
    dm_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    dm_kappas: tuple[float, ...] = (20.0, 20.0, 20.0)

    # Cell loss configuration
    cell_loss_mode: str = "balanced"  # "balanced" | "mass_weighted"
    cell_beta: float = 1.0
    cell_mass_weight_eps: float = 1e-3
    cell_mass_weight_alpha: float = 1.0  # relative crowd boost
    cell_mass_weight_gamma: float = 1.0  # exponent on normalized crowd mass distribution
    lambda_curvature: float = 0.0        # weight for curvature power loss (0 = disabled)
    lambda_hard_bg: float = 0.0          # weight for top-k hard background loss (0 = disabled)
    hard_bg_ratio: float = 0.10          # fraction of worst false alarm background pixels to penalize
    lambda_fg_gate: float = 0.0          # weight for foreground gate BCE loss (0 = disabled)

    # Dual-Lattice Carrier Supervision (RMR-v30 H2/H3: subpixel_stride2=True)
    lambda_carrier_cell: float = 0.0    # carrier (Stride 4) mass-weighted cell loss weight
    lambda_fine_cell: float = 0.0       # fine (Stride 2) mass-weighted cell loss weight

    # RMR-v12 Density-Gated Curvature additions
    curvature_gate_threshold: float = 0.0  # density threshold to activate curvature loss (0 = disabled / full image)
    curvature_gate_kernel: int = 5         # kernel size for local density pooling (covers 20x20 px at stride 4)
    curvature_gate_mode: str = "none"      # "none" | "hard" | "soft"
    curvature_gate_scale: float = 0.02     # temperature for soft sigmoid transition

    # RMR-v13 Physical Scale Alignment Loss additions
    lambda_scale_align: float = 0.0      # weight for physical scale alignment loss (0 = disabled)
    scale_align_tau_dense: float = 0.12  # local density threshold for fine scale (16x16)
    scale_align_tau_sparse: float = 0.03 # local density threshold for coarse scale (64x64)
    scale_align_kernel: int = 5          # kernel size for local density estimation
    scale_align_mask_bg: bool = True     # RMR-v14: mask out background pixels from scale alignment KL loss

    # RMR-v20 High-Density Sample-Level Loss Scaling (0 params)
    density_loss_scaling: bool = False
    dense_loss_thresh: float = 100.0
    dense_loss_norm: float = 150.0
    dense_loss_alpha: float = 1.0
    dense_loss_max_boost: float = 2.0

    # RMR-v21 Elementwise Sample-Level Loss Scaling (0 params)
    elementwise_dense_scaling: bool = False

    # Spectral Loss configuration (Hypothesis H2)
    use_spectral_loss: bool = False
    lambda_spectral: float = 0.0
    spectral_beta: float = 2.0
    lambda_spectral_dc: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_spectral < 0.0:
            raise ValueError(f"lambda_spectral must be non-negative, got {self.lambda_spectral}")
        if self.lambda_spectral_dc < 0.0:
            raise ValueError(f"lambda_spectral_dc must be non-negative, got {self.lambda_spectral_dc}")
        if self.spectral_beta <= 0.0:
            raise ValueError(f"spectral_beta must be strictly positive, got {self.spectral_beta}")
        if self.use_hierarchical_dm and not self.use_multiscale_dm:
            self.use_multiscale_dm = True
        if self.dm_target not in ("y", "y0", "dual"):
            raise ValueError(f"dm_target must be 'y', 'y0', or 'dual', got '{self.dm_target}'")
        if self.count_loss_mode not in ("nb", "log1p", "l1"):
            raise ValueError(f"count_loss_mode must be 'nb', 'log1p', or 'l1', got '{self.count_loss_mode}'")
        if self.cell_loss_mode not in ("balanced", "mass_weighted"):
            raise ValueError(f"cell_loss_mode must be 'balanced' or 'mass_weighted', got '{self.cell_loss_mode}'")
        if self.allocation_loss_type not in ("flat_dm16", "bayesian", "ot_sinkhorn"):
            raise ValueError(f"allocation_loss_type must be 'flat_dm16', 'bayesian', or 'ot_sinkhorn', got '{self.allocation_loss_type}'")
        if self.cell_mass_weight_gamma <= 0.0:
            raise ValueError(f"cell_mass_weight_gamma must be strictly positive, got {self.cell_mass_weight_gamma}")
        if self.curvature_gate_mode not in ("none", "hard", "soft"):
            raise ValueError(f"curvature_gate_mode must be 'none', 'hard', or 'soft', got '{self.curvature_gate_mode}'")
        if self.lambda_curvature < 0.0:
            raise ValueError(f"lambda_curvature must be non-negative, got {self.lambda_curvature}")
        if self.lambda_hard_bg < 0.0:
            raise ValueError(f"lambda_hard_bg must be non-negative, got {self.lambda_hard_bg}")
        if not (0.0 < self.hard_bg_ratio <= 1.0):
            raise ValueError(f"hard_bg_ratio must be in (0.0, 1.0], got {self.hard_bg_ratio}")
        if self.lambda_fg_gate < 0.0:
            raise ValueError(f"lambda_fg_gate must be non-negative, got {self.lambda_fg_gate}")
        if self.lambda_scale_align < 0.0:
            raise ValueError(f"lambda_scale_align must be non-negative, got {self.lambda_scale_align}")
        if self.lambda_carrier_cell < 0.0:
            raise ValueError(f"lambda_carrier_cell must be non-negative, got {self.lambda_carrier_cell}")
        if self.lambda_fine_cell < 0.0:
            raise ValueError(f"lambda_fine_cell must be non-negative, got {self.lambda_fine_cell}")
        if self.scale_align_tau_dense <= self.scale_align_tau_sparse:
            raise ValueError(
                f"scale_align_tau_dense ({self.scale_align_tau_dense}) must be > scale_align_tau_sparse ({self.scale_align_tau_sparse})"
            )
        if self.scale_align_kernel <= 0 or self.scale_align_kernel % 2 == 0:
            raise ValueError(
                f"scale_align_kernel must be a positive odd integer, got {self.scale_align_kernel}"
            )


    @classmethod
    def from_dict(cls, d: dict | None) -> "RMRv3LossConfig":
        if not d:
            return cls()

        field_map = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: dict = {}
        use_multi = bool(d.get("use_multiscale_dm", d.get("use_hierarchical_dm", False)))
        kwargs["use_multiscale_dm"] = use_multi
        kwargs["use_hierarchical_dm"] = use_multi

        for k, v in d.items():
            if k in ("use_multiscale_dm", "use_hierarchical_dm"):
                continue
            if k not in field_map or k.startswith("_"):
                continue
            field = field_map[k]
            default_val = (
                field.default
                if field.default is not dataclasses.MISSING
                else (field.default_factory() if field.default_factory is not dataclasses.MISSING else None)
            )
            if isinstance(default_val, bool):
                kwargs[k] = bool(v)
            elif isinstance(default_val, int):
                kwargs[k] = int(v)
            elif isinstance(default_val, float):
                kwargs[k] = float(v)
            elif isinstance(default_val, str):
                kwargs[k] = str(v)
            elif isinstance(default_val, tuple) and isinstance(v, (list, tuple)):
                kwargs[k] = tuple(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)
