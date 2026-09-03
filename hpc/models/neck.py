import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated


class ScaleGradient(torch.autograd.Function):
    """Autograd function that passes input through unchanged on forward, but scales gradient on backward."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale == 0.0:
            return torch.zeros_like(grad_output), None
        if ctx.scale == 1.0:
            return grad_output, None
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return ScaleGradient.apply(x, scale)


class AdditiveFPNNeck(nn.Module):
    """Additive depthwise-separable FPN with optional C32/P32 carrier.

    Channels: width (default C=32)
    Inputs: c4, c8, c16 and optionally c32 from the backbone
    Output: p4 feature map at reduction 4 with width channels
    """

    def __init__(
        self,
        in_channels: tuple = (24, 48, 80),
        width: int = 32,
        context_dilations: tuple = (1, 2, 3),
        use_p8_context: bool = False,
        c32_grad_scale: float = 1.0,
        target_stride: int | None = None,
    ):
        super().__init__()
        if len(in_channels) not in {3, 4}:
            raise ValueError(f"AdditiveFPNNeck expects 3 or 4 input levels, got {len(in_channels)}")
        c4, c8, c16 = in_channels[:3]
        self.width = width
        self.use_p8_context = use_p8_context
        self.use_p32 = len(in_channels) == 4
        self.c32_grad_scale = float(c32_grad_scale)
        self.target_stride = target_stride

        # P16 path (base level of top-down FPN)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)
        if self.use_p32:
            self.lat32 = ConvGNAct(in_channels[3], width, kernel_size=1)
            self.ref32 = DSResidual(width)

        # Multi-dilation coarse context at reduction 16
        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])
        self.ref16 = DSResidual(width)

        # P8 path: only instantiated if target_stride in {4, 8} or None
        if target_stride is None or target_stride in {4, 8}:
            self.lat8 = ConvGNAct(c8, width, kernel_size=1)
            self.ref8 = DSResidual(width)
            if self.use_p8_context:
                self.context_p8_blocks = nn.ModuleList([
                    DepthwiseDilated(width, dilation=d) for d in (1, 2, 3)
                ])
                self.context_p8_fuse = ConvGNAct(width, width, kernel_size=1)
                nn.init.zeros_(self.context_p8_fuse.conv.weight)
        else:
            self.lat8 = None
            self.ref8 = None
            self.context_p8_blocks = None
            self.context_p8_fuse = None

        # P4 path: only instantiated if target_stride == 4 or None
        if target_stride is None or target_stride == 4:
            self.lat4 = ConvGNAct(c4, width, kernel_size=1)
            self.ref4 = DSResidual(width)
        else:
            self.lat4 = None
            self.ref4 = None

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
        c32: torch.Tensor | None = None,
        return_routes: bool = False,
        target_stride: int | None = None,
    ):
        stride = target_stride if target_stride is not None else self.target_stride

        if self.use_p32 != (c32 is not None):
            expected = "four (C4/C8/C16/C32)" if self.use_p32 else "three (C4/C8/C16)"
            raise ValueError(f"Neck was constructed for {expected} feature tensors")

        # 1. P16 computation
        l16 = self.lat16(c16)
        p32 = None
        p16_in = l16
        if self.use_p32:
            c32_in = scale_gradient(c32, self.c32_grad_scale)
            p32 = self.ref32(self.lat32(c32_in))
            p16_in = p16_in + F.interpolate(
                p32, size=l16.shape[-2:], mode="bilinear", align_corners=False
            )

        ctx_sum = sum(ctx(p16_in) for ctx in self.context_blocks) if len(self.context_blocks) > 0 else 0
        p16 = self.ref16(p16_in + ctx_sum)

        if stride == 16:
            if return_routes:
                routes = {"p16": p16}
                if p32 is not None:
                    routes["p32"] = p32
                return p16, routes
            return p16

        # 2. P8 computation
        if self.lat8 is None or self.ref8 is None:
            raise RuntimeError(
                f"Neck was initialized with target_stride={self.target_stride}, "
                f"cannot compute stride {stride}"
            )
        l8 = self.lat8(c8)
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8_in = l8 + up16_to_8
        if self.use_p8_context and self.context_p8_blocks is not None:
            ctx_sum8 = sum(ctx(p8_in) for ctx in self.context_p8_blocks)
            p8_in = p8_in + self.context_p8_fuse(ctx_sum8)

        p8 = self.ref8(p8_in)

        if stride == 8:
            if return_routes:
                routes = {
                    "p8": p8,
                    "p16": p16,
                }
                if p32 is not None:
                    routes["p32"] = p32
                return p8, routes
            return p8

        # 3. P4 computation
        if self.lat4 is None or self.ref4 is None:
            raise RuntimeError(
                f"Neck was initialized with target_stride={self.target_stride}, "
                f"cannot compute stride {stride}"
            )
        l4 = self.lat4(c4)
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        if return_routes:
            routes = {
                "p4": p4,
                "p8": p8,
                "p16": p16,
            }
            if p32 is not None:
                routes["p32"] = p32
            return p4, routes
        return p4
