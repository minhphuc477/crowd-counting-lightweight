import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .backbone import MobileNetV4Backbone
from .neck import AdditiveFPNNeck


def inv_softplus(y: float) -> float:
    """Numerically stable inverse of softplus for y > 0.

    softplus^{-1}(y) = log(exp(y) - 1)
                       = y + log(1 - exp(-y)).
    The second form avoids overflow for large y.
    """
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y))


class HPCLite(nn.Module):
    """HPC-Lite lightweight crowd counter.

    Image -> MobileNetV4 features at reductions 4/8/16 -> 32-ch additive FPN
          -> 1-channel positive count-mass map D at stride 4 -> sum(D).
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = True,
        neck_width: int = 32,
        context_dilations: Tuple[int, ...] = (1, 2, 3),
        eps_d: float = 1e-6,
        truncate_backbone: bool = True,
        output_stride: int = 4,
    ):
        super().__init__()
        self.eps_d = float(eps_d)
        self.neck_width = int(neck_width)
        self.output_stride = int(output_stride)
        if self.output_stride != 4:
            raise ValueError("Current neck/targets assume output_stride=4")

        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            target_reductions=(4, 8, 16),
            truncate=truncate_backbone,
        )

        self.neck = AdditiveFPNNeck(
            in_channels=self.backbone.out_channels,
            width=neck_width,
            context_dilations=context_dilations,
        )

        self.head_dw = nn.Conv2d(
            neck_width,
            neck_width,
            kernel_size=3,
            padding=1,
            groups=neck_width,
            bias=False,
        )
        num_groups = min(8, neck_width)
        while neck_width % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.head_norm = nn.GroupNorm(num_groups, neck_width)
        self.head_act = nn.SiLU(inplace=True)
        self.head_out = nn.Conv2d(neck_width, 1, kernel_size=1, bias=True)

        nn.init.constant_(self.head_out.bias, -6.0)
        nn.init.normal_(self.head_out.weight, std=0.01)

    def init_head_bias_from_data(
        self,
        mean_crop_count: float,
        crop_size: int,
        output_stride: int = 4,
    ) -> None:
        """Initialize output bias from the expected mean mass per output cell."""
        grid_h = math.ceil(crop_size / output_stride)
        grid_w = math.ceil(crop_size / output_stride)
        n_cells = max(grid_h * grid_w, 1)
        m0 = max(float(mean_crop_count) / n_cells, 1e-8)
        with torch.no_grad():
            nn.init.constant_(self.head_out.bias, inv_softplus(m0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B,3,H,W), got {tuple(x.shape)}")
        c4, c8, c16 = self.backbone(x)
        p4 = self.neck(c4, c8, c16)
        h = self.head_act(self.head_norm(self.head_dw(p4)))
        z = self.head_out(h)
        return F.softplus(z) + self.eps_d

    @torch.no_grad()
    def predict(self, x: torch.Tensor, pad_multiple: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variable-resolution single-scale inference.

        Pads only on right/bottom, then keeps ceil(H/4) x ceil(W/4) output cells.
        Using floor(H/4) drops valid border content for odd image dimensions.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got {tuple(x.shape)}")
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be > 0")

        _, _, h, w = x.shape
        pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
        pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple

        if pad_h > 0 or pad_w > 0:
            # reflect requires padding smaller than the corresponding input dimension.
            mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        d_padded = self.forward(x)
        out_h = math.ceil(h / self.output_stride)
        out_w = math.ceil(w / self.output_stride)
        if d_padded.shape[-2] < out_h or d_padded.shape[-1] < out_w:
            raise RuntimeError(
                f"Backbone output {tuple(d_padded.shape[-2:])} is smaller than required "
                f"valid map {(out_h, out_w)}"
            )

        d_valid = d_padded[..., :out_h, :out_w]
        count = d_valid.sum(dim=(-1, -2, -3))
        return count, d_valid
