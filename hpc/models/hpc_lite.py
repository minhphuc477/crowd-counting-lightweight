import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .backbone import ShuffleNetV2PyramidBackbone
from .neck import ScaleRoutedFusionNeck
from .blocks import make_group_norm


def inv_softplus(y: float) -> float:
    """Numerically stable inverse of softplus for y > 0.

    softplus^{-1}(y) = log(exp(y) - 1)
                     = y + log(1 - exp(-y))    [second form avoids overflow for large y]
    """
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y))


class HPCLiteSR48(nn.Module):
    """HPC-S Scale-Routed crowd counter, ~0.175M deploy parameters.

    Architecture:
      Image → ShuffleNetV2 ×0.5 backbone (C4/C8/C16/C32)
            → ScaleRoutedFusionNeck (48ch, /4)
            → DW5×5 + GN + SiLU mass head
            → D = Softplus(z) + ε  (stride-4 count-mass map)
            → Count = sum(D)

    Deploy graph excludes all training-only modules (NB dispersion, targets, masks, KD).
    """

    def __init__(
        self,
        pretrained: bool = True,
        neck_width: int = 48,
        eps_d: float = 1e-6,
        route_temperature: float = 1.0,
        pool_kernels=(3, 5, 7),
        pool_residual_mix: float = 0.5,
        simam_lambda: float = 1e-4,
        output_stride: int = 4,
    ):
        super().__init__()
        if output_stride != 4:
            raise ValueError("HPC targets assume output_stride=4")
        self.output_stride = output_stride
        self.eps_d = float(eps_d)
        self.neck_width = int(neck_width)

        self.backbone = ShuffleNetV2PyramidBackbone(pretrained=pretrained)
        self.neck = ScaleRoutedFusionNeck(
            in_channels=self.backbone.out_channels,  # [24, 48, 96, 192]
            width=neck_width,
            route_temperature=route_temperature,
            pool_kernels=pool_kernels,
            pool_residual_mix=pool_residual_mix,
            simam_lambda=simam_lambda,
        )

        # DW5×5 + GN + SiLU + PW1×1 count-mass head
        self.head_dw = nn.Conv2d(
            neck_width,
            neck_width,
            kernel_size=5,
            padding=2,
            groups=neck_width,
            bias=False,
        )
        self.head_norm = make_group_norm(neck_width)
        self.head_act = nn.SiLU(inplace=True)
        self.head_out = nn.Conv2d(neck_width, 1, kernel_size=1, bias=True)

        # Conservative init: expect near-zero output counts initially
        nn.init.constant_(self.head_out.bias, -6.0)
        nn.init.normal_(self.head_out.weight, std=0.01)

    def init_head_bias_from_data(
        self,
        mean_crop_count: float,
        crop_size: int,
        output_stride: int = 4,
    ) -> None:
        """Data-driven head bias initialization using stable inverse softplus."""
        grid_h = math.ceil(crop_size / output_stride)
        grid_w = math.ceil(crop_size / output_stride)
        n_cells = max(grid_h * grid_w, 1)
        m0 = max(float(mean_crop_count) / n_cells, 1e-8)
        # Cap before inv_softplus to avoid overflow (spec §8.2)
        m0 = min(m0, 1e4)
        with torch.no_grad():
            nn.init.constant_(self.head_out.bias, inv_softplus(m0))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image tensor (ImageNet-normalized).
            return_aux: if True, also return route diagnostics dict.

        Returns:
            d: (B, 1, H/4, W/4) positive count-mass density map.
            aux: dict with routing diagnostics (only when return_aux=True).
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B, 3, H, W), got {tuple(x.shape)}")

        c4, c8, c16, c32 = self.backbone(x)

        if return_aux:
            p4, aux = self.neck(c4, c8, c16, c32, return_routes=True)
        else:
            p4 = self.neck(c4, c8, c16, c32)

        h = self.head_act(self.head_norm(self.head_dw(p4)))
        d = F.softplus(self.head_out(h)) + self.eps_d

        return (d, aux) if return_aux else d

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        pad_multiple: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variable-resolution single-scale inference.

        Pads right/bottom to multiple of 32 (required for /32 backbone stride)
        using zero (neutral ImageNet-normalized) padding.
        Crops output to ceil(H/4) × ceil(W/4) valid cells.

        Returns:
            count: (B,) per-image crowd count.
            d_valid: (B, 1, ceil(H/4), ceil(W/4)) count-mass map.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {tuple(x.shape)}")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be > 0")

        _, _, h, w = x.shape
        pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
        pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple

        if pad_h > 0 or pad_w > 0:
            # Zero padding: neutral for ImageNet-normalized input (approx mean 0)
            # Avoids duplicating border people (reflect/replicate can create false positives)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        d_padded = self.forward(x)

        # Valid output uses ceil (never floor) to preserve border content
        out_h = math.ceil(h / self.output_stride)
        out_w = math.ceil(w / self.output_stride)

        if d_padded.shape[-2] < out_h or d_padded.shape[-1] < out_w:
            raise RuntimeError(
                f"Backbone output {tuple(d_padded.shape[-2:])} smaller than "
                f"required valid map {(out_h, out_w)}"
            )

        d_valid = d_padded[..., :out_h, :out_w]
        count = d_valid.sum(dim=(-1, -2, -3))
        return count, d_valid

    def predict_sliding_window(
        self,
        x: torch.Tensor,
        crop_size: int = 448,
        stride: int = 224,
        use_tta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sliding-window inference with 2D Gaussian blending and Test-Time Augmentation (TTA).

        Solves out-of-distribution scaling and boundary artifacts on large test images.

        Args:
            x: (1, 3, H, W) input image tensor.
            crop_size: window size (matches training crop resolution).
            stride: sliding step size (overlap = crop_size - stride).
            use_tta: if True, averages predictions across original and horizontally flipped views.

        Returns:
            count: (1,) scalar count.
            d_global: (1, 1, ceil(H/4), ceil(W/4)) seamlessly blended density map.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected single image (1, 3, H, W), got {tuple(x.shape)}")

        _, _, H, W = x.shape
        out_h = math.ceil(H / self.output_stride)
        out_w = math.ceil(W / self.output_stride)

        if H <= crop_size and W <= crop_size:
            if use_tta:
                c1, d1 = self.predict(x)
                x_flip = torch.flip(x, dims=[-1])
                c2, d2 = self.predict(x_flip)
                d2 = torch.flip(d2, dims=[-1])
                d_avg = 0.5 * (d1 + d2)
                return d_avg.sum(dim=(-1, -2, -3)), d_avg
            return self.predict(x)

        # 2D Gaussian blending window for smooth overlap transitions
        cw4 = crop_size // self.output_stride
        sigma = cw4 / 4.0
        y_axis = torch.arange(cw4, dtype=torch.float32, device=x.device) - (cw4 - 1) / 2.0
        x_axis = torch.arange(cw4, dtype=torch.float32, device=x.device) - (cw4 - 1) / 2.0
        gy, gx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        kernel = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2)).unsqueeze(0).unsqueeze(0)  # (1, 1, cw4, cw4)

        d_accum = torch.zeros(1, 1, out_h, out_w, dtype=torch.float32, device=x.device)
        w_accum = torch.zeros(1, 1, out_h, out_w, dtype=torch.float32, device=x.device)

        y_steps = list(range(0, max(1, H - crop_size + 1), stride))
        if y_steps[-1] != H - crop_size and H > crop_size:
            y_steps.append(H - crop_size)

        x_steps = list(range(0, max(1, W - crop_size + 1), stride))
        if x_steps[-1] != W - crop_size and W > crop_size:
            x_steps.append(W - crop_size)

        for y0 in y_steps:
            for x0 in x_steps:
                y1 = min(H, y0 + crop_size)
                x1 = min(W, x0 + crop_size)
                y0_adj = max(0, y1 - crop_size)
                x0_adj = max(0, x1 - crop_size)

                patch = x[:, :, y0_adj:y1, x0_adj:x1]

                if use_tta:
                    _, dp1 = self.predict(patch)
                    patch_flip = torch.flip(patch, dims=[-1])
                    _, dp2 = self.predict(patch_flip)
                    dp2 = torch.flip(dp2, dims=[-1])
                    dp = 0.5 * (dp1 + dp2)
                else:
                    _, dp = self.predict(patch)

                # Output grid coordinates
                out_y0 = y0_adj // self.output_stride
                out_x0 = x0_adj // self.output_stride
                ph, pw = dp.shape[-2], dp.shape[-1]

                k = kernel[:, :, :ph, :pw]
                d_accum[:, :, out_y0:out_y0+ph, out_x0:out_x0+pw] += dp * k
                w_accum[:, :, out_y0:out_y0+ph, out_x0:out_x0+pw] += k

        d_global = d_accum / (w_accum + 1e-8)
        count = d_global.sum(dim=(-1, -2, -3))
        return count, d_global
