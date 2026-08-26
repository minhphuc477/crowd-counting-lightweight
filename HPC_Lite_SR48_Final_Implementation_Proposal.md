# HPC-Lite-SR48 — Final Research & Implementation Proposal

**Status:** authoritative implementation handoff for the next coding agent  
**Date:** 2026-08-26  
**Primary goal:** replace the current ~0.350M deployment model with a scale-routed ~0.175M student while preserving HPC-Lite's probabilistic counting formulation and directly addressing the observed failure on **large/near-camera people and people close to true image borders**.

---

## 0. Read this first

This document is intended to be sufficient for another agent to modify the code without having to reconstruct design decisions from chat history.

### Source-of-truth order

1. **This file** for the new SR48 architecture and migration plan.
2. `HPC_Lite_fixed_files.zip` / `HPC_Lite_bugfix_apply.patch` for the previously audited bug fixes.
3. `HPC_Lite_Final_Agent_Spec.md` for the original HPC-Lite mathematical definition where this file does not explicitly replace it.
4. The raw uploaded `.py` files are **not** the preferred base if they still contain pre-audit bugs.

### Mandatory rule

**Do not overwrite the old `HPCLite` model.** Keep it as the 0.350M baseline/ablation and add the new model as a separate class/config, e.g. `HPCLiteSR48`.

### Previously fixed bugs that must not regress

The implementation must preserve these fixes:

- numerically stable inverse softplus;
- curriculum weights must multiply configured lambdas so `lambda=0` is a true ablation at every phase;
- criterion learnable dispersion parameters must be moved to device, optimized, saved, and restored;
- hard-negative/empty targets must be on the same device as predictions;
- variable-resolution prediction must use **ceil** output size, not floor;
- strict dataset annotation loading: parse/missing GT must not silently become empty scenes;
- geometric resize must be isotropic;
- second-view batches must have a deterministic collatable schema;
- NWPU NAE must exclude zero-GT images and divide by `gt`, not `gt+1`;
- prediction/GT metric arrays must have equal shapes;
- CSV logging must not silently discard later fields.

---

# 1. Research motivation

## 1.1 Current model

Current deploy graph:

```text
Image
  ↓
MobileNetV4-Conv-Small-0.5, truncated at /16
  ↓
C4 /4, C8 /8, C16 /16
  ↓
32-channel additive FPN
  ↓
parallel DW3×3 dilation {1,2,3} at /16
  ↓
DS residual refinements
  ↓
DW3×3 + PW1×1 mass head
  ↓
Softplus count-mass map D @ stride 4
  ↓
sum(D)
```

Profile reported by the current code:

\[
\boxed{350,017\ \text{deploy parameters}}
\]

The model counts distant/small people relatively well, but misses a nontrivial fraction of people who are:

- close to camera / visually large;
- isolated or sparse;
- very close to a true image edge;
- partially visible at a border.

This strongly suggests **scale-selection bias plus border/context bias**, not simply inadequate total parameter count.

## 1.2 Why not just add more RCMB/attention layers?

LRMBNet V0.5 reports roughly:

\[
0.25M\ \text{params},\qquad \text{SHA MAE}=59.94,
\]

and uses aggressive channel compression followed by multi-scale residual multi-branch processing. It demonstrates that a small parameter count can still support substantial spatial computation.

However, directly appending LRMBNet-style multi-branch blocks to HPC-Lite would weaken the paper's parameter-efficiency story and does not specifically solve the observed scale-routing failure.

## 1.3 Research evidence motivating SR48

The new design borrows *principles*, not architecture copies:

- **Context-Aware Crowd Counting (CVPR 2019):** learns the importance of multiple receptive-field sizes at each spatial location; perspective variation requires location-dependent context.
- **STEERER (ICCV 2023):** argues indiscriminate cooperative multi-resolution fusion can be suboptimal and explicitly selects/inherits scale-customized features.
- **SimAM (ICML 2021):** parameter-free 3D feature attention.
- **PoolFormer / MetaFormer (CVPR 2022):** non-parametric pooling can perform useful spatial token mixing.
- **HybridCount (Pattern Recognition 2026):** reports a very small student (`0.22M`) with strong SHT-A accuracy (`52.25 MAE`) using scale-aware knowledge distillation, showing that tiny deploy models can benefit from stronger training-time supervision.
- **ReviewKD for crowd counting:** supports training-time distillation as a way to improve lightweight students without increasing deploy parameters.

### Final design principle

> **Spend parameters on a strong but tiny backbone and explicit per-location scale routing; spend training complexity on supervision/distillation; avoid expensive deploy-only heads.**

---

# 2. Final model recommendation

## 2.1 Name

**HPC-Lite-SR48**  
SR = **Scale-Routed**, width = 48.

## 2.2 Target deployment budget

Analytical parameter estimate:

\[
\boxed{174,629\ \text{parameters}\approx0.175M}
\]

This is about:

\[
\boxed{50\%\ \text{fewer parameters than current HPC-Lite}}
\]

and about 30% fewer than LRMBNet V0.5's reported 0.25M.

**Do not publish the analytical number as final until the implementation's `sum(p.numel())` confirms it.**

## 2.3 Accuracy targets

Minimum target for ShanghaiTech Part A:

\[
\boxed{\text{MAE}<59.94}
\]

Preferred target:

\[
\boxed{53\text{–}57}
\]

The research objective is Pareto efficiency, not parameter count alone.

---

# 3. Deployment architecture

## 3.1 Full graph

```text
Input image
   │
   ▼
ShuffleNetV2 ×0.5 ImageNet-pretrained feature backbone
   │
   ├────────────── C4   /4   24 ch
   ├────────────── C8   /8   48 ch
   ├────────────── C16 /16   96 ch
   └────────────── C32 /32  192 ch
                    │
                    ▼
         Four lateral 1×1 projections
          24/48/96/192 → 48 ch
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
      R4 /4       R8 /8      R16 /16
     DS-Res       DS-Res       DS-Res
                                │
                                │       R32 /32
                                │       zero-param
                                │       multi-pool context
                                │       + DS-Res
        └──────────────┬───────────────┘
                       │
                       ▼
             Route features at /8
   [area-down R4, R8, ↑R16, ↑R32] concat
                       │
                Conv1×1 192→4
                       │
                  softmax(scale)
                       │
            α4, α8, α16, α32 @ /8
                       │
                 ↑ weights to /4
                       │
                       ▼
   F = α4 R4 + α8 ↑R8 + α16 ↑R16 + α32 ↑R32
                       │
                       ▼
              SimAM (0 parameters)
                       │
                       ▼
           DW5×5 + GN + SiLU
                       │
                       ▼
                 Conv1×1 → 1
                       │
                       ▼
             D = Softplus(z)+1e-6
                 stride 4 mass map
                       │
                       ▼
                 Count = sum(D)
```

## 3.2 Why four scales

The old network stops at `/16`. The new student deliberately keeps `/32` because large/near-camera people require broader semantic/receptive-field support.

The backbone remains tiny enough that `/32` can be retained while **halving total model parameters**.

## 3.3 Tensor shapes

### For 448×448 crop

| Tensor | Shape |
|---|---|
| input | `B×3×448×448` |
| C4 | `B×24×112×112` |
| C8 | `B×48×56×56` |
| C16 | `B×96×28×28` |
| C32 | `B×192×14×14` |
| R4 | `B×48×112×112` |
| R8 | `B×48×56×56` |
| R16 | `B×48×28×28` |
| R32 | `B×48×14×14` |
| router logits | `B×4×56×56` |
| fused | `B×48×112×112` |
| D | `B×1×112×112` |

### For 672×672 crop

| Tensor | Shape |
|---|---|
| input | `B×3×672×672` |
| C4 | `B×24×168×168` |
| C8 | `B×48×84×84` |
| C16 | `B×96×42×42` |
| C32 | approximately `B×192×21×21` |
| fused | `B×48×168×168` |
| D | `B×1×168×168` |

Use explicit interpolation `size=target.shape[-2:]`; never assume exact `×2` for arbitrary image dimensions.

---

# 4. Exact parameter budget

PyTorch ShuffleNetV2 ×0.5 standard components:

| Component | Params |
|---|---:|
| conv1 | 696 |
| stage2 | 6,936 |
| stage3 | 45,552 |
| stage4 | 89,952 |
| **feature backbone through stage4** | **143,136** |
| conv5 — discarded | 198,656 |
| classifier — discarded | 1,025,000 |

SR48 additions:

| Component | Params |
|---|---:|
| backbone through `/32` | 143,136 |
| four lateral Conv+GN blocks | 17,664 |
| four 48ch DSResidual blocks | 11,712 |
| scale router `192→4` | 772 |
| zero-param pool context | 0 |
| SimAM | 0 |
| DW5×5 + GN + 1×1 mass head | 1,345 |
| **TOTAL** | **174,629** |

The exact count assumes:

- lateral 1×1 convs have `bias=False`;
- GroupNorm has affine weight+bias;
- router conv has bias;
- head final 1×1 has bias.

### Neck/head compute estimate only

Analytical MACs excluding backbone/interpolation/elementwise operations:

- ~90.7M MACs at 448×448;
- ~204.2M MACs at 672×672.

Use a profiler for the final end-to-end number and clearly distinguish **MACs** from **FLOPs**.

---

# 5. Backbone implementation

## 5.1 File

`models/backbone.py`

Keep `MobileNetV4Backbone` for baseline. Add:

```python
from torchvision.models import (
    shufflenet_v2_x0_5,
    ShuffleNet_V2_X0_5_Weights,
)


class ShuffleNetV2PyramidBackbone(nn.Module):
    """Return spatial features at reductions 4, 8, 16, 32.

    Intentionally excludes conv5, global pooling, and classifier.
    Expected channels: [24, 48, 96, 192].
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        base = shufflenet_v2_x0_5(weights=weights)

        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4

        self.out_channels = [24, 48, 96, 192]
        self.out_reductions = [4, 8, 16, 32]

    def forward(self, x):
        x = self.conv1(x)       # /2
        c4 = self.maxpool(x)    # /4, 24
        c8 = self.stage2(c4)    # /8, 48
        c16 = self.stage3(c8)   # /16, 96
        c32 = self.stage4(c16)  # /32, 192
        return c4, c8, c16, c32
```

## 5.2 Required tests

```python
m = ShuffleNetV2PyramidBackbone(pretrained=False)
x = torch.randn(2, 3, 448, 448)
f = m(x)
assert [z.shape[1] for z in f] == [24, 48, 96, 192]
assert sum(p.numel() for p in m.parameters()) == 143136
```

Also test 672×672 and odd resolution such as 449×451.

---

# 6. New blocks

## 6.1 File

`models/blocks.py`

Preserve existing `ConvGNAct`, `DSResidual`. Add the following.

## 6.2 Parameter-free multi-pool context

```python
class MultiPoolContext(nn.Module):
    """Parameter-free coarse spatial mixer.

    Residual mixing prevents pure average-pool oversmoothing.
    """

    def __init__(self, kernels=(3, 5, 7), residual_mix: float = 0.5):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.residual_mix = float(residual_mix)
        for k in self.kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError("pool kernels must be positive odd integers")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0,1]")

    def forward(self, x):
        if not self.kernels:
            return x
        pooled = [
            F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            for k in self.kernels
        ]
        p = torch.stack(pooled, dim=0).mean(dim=0)
        return x + self.residual_mix * (p - x)
```

Default `residual_mix=0.5` is a constant, not learnable.

At `/32`, kernels `{3,5,7}` expose very broad input-space context while adding zero parameters.

## 6.3 SimAM

Use the paper's parameter-free formulation; do not accidentally turn `lambda_e` into an `nn.Parameter`.

```python
class SimAM(nn.Module):
    def __init__(self, lambda_e: float = 1e-4):
        super().__init__()
        self.lambda_e = float(lambda_e)

    def forward(self, x):
        h, w = x.shape[-2:]
        n = h * w - 1
        if n <= 0:
            return x
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / float(n)
        e_inv = d / (4.0 * (v + self.lambda_e)) + 0.5
        return x * torch.sigmoid(e_inv)
```

### Test

```python
mod = SimAM()
assert sum(p.numel() for p in mod.parameters()) == 0
```

---

# 7. Scale-Routed fusion neck

## 7.1 File

`models/neck.py`

Keep `AdditiveFPNNeck` as baseline. Add `ScaleRoutedFusionNeck`.

## 7.2 Core equations

Let the four backbone features be:

\[
C_s,\quad s\in\{4,8,16,32\}.
\]

Project each to 48 channels:

\[
L_s=\phi_s(C_s),\qquad L_s\in\mathbb{R}^{48\times H/s\times W/s}.
\]

Refine independently:

\[
R_4=\mathcal R_4(L_4),\quad
R_8=\mathcal R_8(L_8),\quad
R_{16}=\mathcal R_{16}(L_{16}),
\]

\[
R_{32}=\mathcal R_{32}(\mathcal P(L_{32})),
\]

where `P` is the zero-parameter pool context.

### Router resolution

Compute routing at `/8` to reduce cost while preserving local perspective changes:

\[
G=
\operatorname{Concat}
\left(
\downarrow R_4,
R_8,
\uparrow R_{16},
\uparrow R_{32}
\right).
\]

\[
A=\operatorname{Softmax}(W_r*G/\tau),
\]

where:

\[
A\in\mathbb{R}^{4\times H/8\times W/8}.
\]

Default temperature:

\[
\tau=1.0.
\]

Upsample routing weights to `/4`, then fuse:

\[
F(x,y)=\sum_{s\in\{4,8,16,32\}}\alpha_s(x,y)\tilde R_s(x,y).
\]

This explicitly allows a near-camera region to favor coarse semantic features while a distant dense region favors high-resolution features.

## 7.3 Suggested code

```python
class ScaleRoutedFusionNeck(nn.Module):
    def __init__(
        self,
        in_channels=(24, 48, 96, 192),
        width=48,
        route_temperature=1.0,
        pool_kernels=(3, 5, 7),
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Expected four backbone feature scales")
        self.width = int(width)
        self.route_temperature = float(route_temperature)

        c4, c8, c16, c32 = in_channels
        self.lat4 = ConvGNAct(c4, width, kernel_size=1)
        self.lat8 = ConvGNAct(c8, width, kernel_size=1)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)
        self.lat32 = ConvGNAct(c32, width, kernel_size=1)

        self.ref4 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref16 = DSResidual(width)
        self.context32 = MultiPoolContext(pool_kernels, residual_mix=0.5)
        self.ref32 = DSResidual(width)

        # 4 × width input → four local scale logits.
        self.router = nn.Conv2d(4 * width, 4, kernel_size=1, bias=True)
        self.attn = SimAM(lambda_e=1e-4)

    @staticmethod
    def _resize(x, size, mode="bilinear"):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def forward(self, c4, c8, c16, c32, return_routes=False):
        r4 = self.ref4(self.lat4(c4))
        r8 = self.ref8(self.lat8(c8))
        r16 = self.ref16(self.lat16(c16))
        r32 = self.ref32(self.context32(self.lat32(c32)))

        # Route at /8. 'area' is deterministic and parameter-free for downsampling.
        g4 = F.interpolate(r4, size=r8.shape[-2:], mode="area")
        g8 = r8
        g16 = self._resize(r16, r8.shape[-2:])
        g32 = self._resize(r32, r8.shape[-2:])

        logits = self.router(torch.cat([g4, g8, g16, g32], dim=1))
        routes8 = torch.softmax(logits / self.route_temperature, dim=1)
        routes4 = self._resize(routes8, r4.shape[-2:])

        u4 = r4
        u8 = self._resize(r8, r4.shape[-2:])
        u16 = self._resize(r16, r4.shape[-2:])
        u32 = self._resize(r32, r4.shape[-2:])

        scales = (u4, u8, u16, u32)
        fused = sum(routes4[:, i:i+1] * scales[i] for i in range(4))
        fused = self.attn(fused)

        if return_routes:
            return fused, {
                "routes8": routes8,
                "routes4": routes4,
                "scale_features": scales,
            }
        return fused
```

### Important

Do **not** use concatenation followed by a wide 48-channel projection after routing. The weighted sum is deliberately cheap.

## 7.4 Router diagnostics

Every validation epoch log:

```text
route_mean_s4
route_mean_s8
route_mean_s16
route_mean_s32
route_entropy
```

Compute:

\[
\bar\alpha_s=\operatorname{mean}_{b,x,y}\alpha_s,
\]

\[
H_{route}=-\operatorname{mean}_{b,x,y}\sum_s\alpha_s\log(\alpha_s+\epsilon).
\]

### Collapse guard

Do **not** add route regularization by default. First observe routing.

Only if one scale receives >90% mean route mass for several epochs, add an optional weak batch-balance penalty:

\[
L_{route-bal}=\sum_s(\bar\alpha_s-0.25)^2
\]

with maximum weight `0.01`.

This must be an ablation, not mandatory core loss.

---

# 8. New model class

## 8.1 File

`models/hpc_lite.py`

Keep `HPCLite`. Add:

```python
class HPCLiteSR48(nn.Module):
    def __init__(
        self,
        pretrained=True,
        neck_width=48,
        eps_d=1e-6,
        route_temperature=1.0,
        output_stride=4,
    ):
        super().__init__()
        if output_stride != 4:
            raise ValueError("HPC targets assume output_stride=4")
        self.output_stride = output_stride
        self.eps_d = float(eps_d)

        self.backbone = ShuffleNetV2PyramidBackbone(pretrained=pretrained)
        self.neck = ScaleRoutedFusionNeck(
            in_channels=self.backbone.out_channels,
            width=neck_width,
            route_temperature=route_temperature,
        )

        self.head_dw = nn.Conv2d(
            neck_width,
            neck_width,
            kernel_size=5,
            padding=2,
            groups=neck_width,
            bias=False,
        )
        self.head_norm = make_group_norm(neck_width)  # or existing GN helper
        self.head_act = nn.SiLU(inplace=True)
        self.head_out = nn.Conv2d(neck_width, 1, kernel_size=1, bias=True)

        nn.init.constant_(self.head_out.bias, -6.0)
        nn.init.normal_(self.head_out.weight, std=0.01)

    def forward(self, x, return_aux=False):
        c4, c8, c16, c32 = self.backbone(x)
        if return_aux:
            p4, aux = self.neck(c4, c8, c16, c32, return_routes=True)
        else:
            p4 = self.neck(c4, c8, c16, c32)
        h = self.head_act(self.head_norm(self.head_dw(p4)))
        d = F.softplus(self.head_out(h)) + self.eps_d
        return (d, aux) if return_aux else d
```

## 8.2 Head initialization

Use the already fixed stable inverse softplus:

\[
m_0=\max\left(\frac{\bar C}{M},10^{-8}\right),
\]

\[
b_0=\operatorname{softplus}^{-1}(m_0).
\]

Do not reintroduce `log(expm1(y))` overflow for large `y`.

## 8.3 Variable-resolution inference

Because the new backbone contains `/32`, default pad multiple must become:

```python
pad_multiple = 32
```

Use normalized-mean zero padding:

```python
x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
```

Reason: reflect/replicate padding can duplicate a person near a true image edge and may create an artificial border pattern. Since normalized ImageNet input has approximate mean zero, zero padding is a neutral default.

Valid output crop:

\[
H_D=\lceil H/4\rceil,\qquad W_D=\lceil W/4\rceil.
\]

Never return `H//4` for odd dimensions.

---

# 9. Target construction — keep the core HPC formulation

The new architecture must **not** change the definition of the deployment output.

\[
D\ge0,\qquad D\in\mathbb R^{1\times H/4\times W/4}
\]

is **count mass per stride-4 cell**, not density per unit area.

Image count:

\[
\hat C=\sum_kD_k.
\]

No pixel-area multiplier.

## 9.1 Exact block counts

For input block size `B`:

\[
b_x=\lfloor x/B\rfloor,\qquad b_y=\lfloor y/B\rfloor.
\]

Direct point histogram only; do not construct block GT by resampling another target.

Scales:

- SHA/SHB 448: `{16,32,64}`;
- QNRF/NWPU 672: `{16,32,96}`.

## 9.2 Block-constrained allocation target

Keep base block `B_A=16`, output stride 4.

Continuous cell coordinates:

\[
u=x/4-1/2,\qquad v=y/4-1/2.
\]

Bilinear splatting is restricted to the point's exact 16×16 block; trim neighbors outside the block and renormalize the surviving weights.

Required invariants:

\[
\sum_{k\in b}Z_{bk}=Y_b^{16},
\]

\[
\sum_kZ_k=N.
\]

Retain the existing tested implementation.

---

# 10. Data-pipeline change: fix the large/border failure

This is **mandatory**. Changing the architecture while leaving the current random crop behavior unchanged can continue teaching the network false negatives at crop boundaries.

## 10.1 Current failure mechanism

A random crop currently:

1. chooses a crop rectangle;
2. subtracts crop origin from points;
3. removes point centers outside the crop.

For a visually large person whose annotated head center is just outside an **artificial crop boundary**, visible human pixels can remain inside the crop while the annotation disappears.

The training pair then becomes:

```text
visible partial person → target zero
```

Repeated exposure can teach the network to suppress exactly the near-camera/border pattern observed in validation.

## 10.2 Required transform class

Modify `datasets/transforms.py`.

Do not replace isotropic scaling. Extend it with **safe candidate crop selection** and metadata.

### Scale proxy

From scaled global point coordinates, compute nearest-neighbor distance:

\[
d_i^{NN}=\min_{j\neq i}\|p_i-p_j\|_2.
\]

For `N=1`, set:

\[
d_i^{NN}=\max(H',W').
\]

Use a crop guard radius:

\[
r_i=\operatorname{clip}(0.20d_i^{NN},8,48)\ \text{pixels}.
\]

These values are defaults for the first experiment, not claimed ground-truth head sizes.

## 10.3 Safe crop condition

For candidate crop:

\[
R=[x_0,x_1)\times[y_0,y_1),
\]

reject it if a point center is outside `R` but lies inside its guard-expanded region.

Conceptually:

```python
inside = (
    (x >= x0) & (x < x1) &
    (y >= y0) & (y < y1)
)
near = (
    (x >= x0 - r) & (x < x1 + r) &
    (y >= y0 - r) & (y < y1 + r)
)
bad = (~inside) & near
safe = not bad.any()
```

This specifically suppresses false-negative partial persons created by **artificial crop edges**.

Points whose center is inside the crop but near its edge remain valid positives; these examples are useful for learning partial-person robustness.

### Sampling attempts

- `max_crop_attempts=20`;
- retain the safe candidate with most desired positives;
- if no safe crop exists, choose the candidate with the smallest number / weighted severity of violating points rather than hanging forever.

## 10.4 Scale-aware crop sampling

Use a mixture rather than pure uniform random crop:

```text
75% safe random crop
15% large/isolated-point-centered crop
10% true-image-border-point-centered crop
```

Initial large candidate condition:

```python
d_nn >= 48 px OR d_nn in top 25% of positive points for the image
```

Initial true border candidate:

\[
d_i^{border}=\min(x_i,y_i,W'-1-x_i,H'-1-y_i)\le32.
\]

A true-image-border point is different from a synthetic crop-border point. True-border examples must be preserved and intentionally sampled.

## 10.5 Return fixed-size special masks, not variable-length lists

Default PyTorch collation is much simpler if the dataset builds fixed maps.

Build after the final crop:

- `gt_large_mask16`: shape `(H/16,W/16)`, 1 for any block containing a large/isolated selected point;
- `gt_true_border_mask16`: same shape, 1 for blocks containing true-image-border points;
- `gt_special_mask16 = max(large,border)`.

These are **training metadata**, not deployment outputs.

### Function suggestion

Add a utility in `targets/special_blocks.py`:

```python
def build_special_block_masks(
    crop_points,
    point_large_flags,
    point_true_border_flags,
    crop_h,
    crop_w,
    block_size=16,
):
    ...
```

Use the same `floor(x/B), floor(y/B)` indexing as exact count targets.

## 10.6 Horizontal flip metadata

If points are flipped, point flags remain attached to the same points; only coordinates change.

## 10.7 Do not use head-size labels that datasets do not provide

`d_nn` is a scale **proxy**, not a true head-size measurement. State that clearly in paper/code comments.

---

# 11. Second photometric view

The fixed dataset schema currently always returns:

```text
image
image_degraded
has_degraded
```

This prevents collate failures.

For the next trainer revision, the preferred semantics are **batch-level robust-view selection** because the original design says roughly 30% of batches. Either of these is acceptable:

### Option A — keep current fixed schema

- every sample has `image_degraded`;
- `has_degraded` masks selected samples;
- robustness loss must operate only on selected samples.

### Option B — preferred later cleanup

- dataset returns only clean image;
- trainer chooses ~30% of batches and applies `PhotometricTransforms` consistently to the batch.

Do not reintroduce optional missing dict keys.

---

# 12. Loss function — final recommended version

## 12.1 Core terms retained

\[
L_{HNB},\quad L_{alloc},\quad L_{HN},\quad L_{empty},\quad L_{global-log},\quad L_{rob}.
\]

Add two zero-deploy-cost terms:

\[
L_{direct},\quad L_{special}.
\]

Optional later:

\[
L_{KD}.
\]

Final:

\[
\boxed{
L=
\lambda_HL_{HNB}
+\lambda_AL_{alloc}
+\lambda_NL_{HN}
+\lambda_EL_{empty}
+\lambda_GL_{global-log}
+\lambda_DL_{direct}
+\lambda_SL_{special}
+\lambda_RL_{rob}
+\lambda_KL_{KD}
}
\]

## 12.2 Hierarchical Negative Binomial

Predicted block mass:

\[
\mu_b^{(B)}=\sum_{k\in b}D_k.
\]

Dispersion:

\[
r_B=\operatorname{softplus}(\eta_B)+10^{-4}.
\]

NLL:

\[
\begin{aligned}
-\log P(Y=y)=&
-\log\Gamma(y+r)+\log\Gamma(r)+\log\Gamma(y+1)\\
&-r\log r-y\log\mu+(r+y)\log(r+\mu).
\end{aligned}
\]

Keep float32 inside NB under AMP.

### Dispersion init

Method of moments:

\[
r_0=\frac{m^2}{v-m},\quad v>m.
\]

If `v<=m`, use large finite approximate-Poisson value.

**Cap `r0`, e.g. at `1e4`, before inverse softplus.**

## 12.3 Density-stratified NB risk

Per scale:

- G0: `y=0`;
- G1: positive `≤q50+`;
- G2: `q50+<y≤q90+`;
- G3: `>q90+`.

Average means of groups present in the batch.

## 12.4 Allocation loss

For each positive 16×16 block:

\[
p_{bk}=\frac{D_{bk}+\epsilon}{\mu_b+K\epsilon},\qquad K=16.
\]

\[
L_{alloc,b}=-\frac1{y_b}\sum_kZ_{bk}\log(p_{bk}+\epsilon).
\]

This loss has a non-zero entropy floor. Add diagnostics:

\[
q_{bk}=Z_{bk}/y_b,
\]

\[
H(q_b)=-\sum_kq_{bk}\log(q_{bk}+\epsilon),
\]

\[
KL(q_b\|p_b)=L_{alloc,b}-H(q_b).
\]

Log:

```text
loss_alloc_ce
alloc_target_entropy
alloc_excess_kl
```

The **KL/excess** is the quantity expected to approach zero, not raw allocation CE.

## 12.5 Special-block weighted allocation

Extend `LocalAllocationLoss.forward`:

```python
forward(d_map, z_map, y16, block_weights=None, return_details=False)
```

For positive block weights:

\[
w_b=1+\beta_sM_b^{special},
\]

initial:

\[
\beta_s=1.0.
\]

Weighted mean:

\[
L_{alloc}=\frac{\sum_bw_bL_{alloc,b}}{\sum_bw_b}.
\]

This makes rare large/border blocks matter more without changing deploy architecture.

## 12.6 Hard negative

On exact zero 16×16 blocks, top 10% predicted masses per image, SmoothL1 to zero.

Validate:

```python
0 < top_fraction <= 1
```

## 12.7 Empty-image loss

For `C_gt=0`:

\[
L_{empty}=\operatorname{mean}(\hat C).
\]

## 12.8 Existing global log loss

\[
L_{global-log}=SmoothL1(\log(1+\hat C),\log(1+C)).
\]

This remains useful for stable dense-count optimization.

## 12.9 New direct count loss

Use the existing `sqrt_normalized` formulation as a second term:

\[
\boxed{
L_{direct}=SmoothL1\left(
\frac{\hat C-C}{\sqrt{C+1}},0
\right)
}
\]

This gives stronger direct count pressure than the log loss while reducing domination by huge-count images.

Implementation can simply instantiate a second `GlobalCountLoss(mode="sqrt_normalized")`; no need to duplicate code.

## 12.10 New special-block count loss

Pool D to exact 16×16 input blocks:

\[
\mu^{16}=PoolSum_{16}(D).
\]

Let:

\[
M^{special}=M^{large}\lor M^{true-border}.
\]

Then:

\[
\boxed{
L_{special}
=
\operatorname{mean}_{M^{special}=1}
SmoothL1(\mu^{16},Y^{16})
}
\]

If no special blocks exist in batch, return exact zero scalar on prediction device.

This is deliberately simple: it boosts count gradients exactly where the observed model fails.

Suggested new class in `losses/hard_negative.py` or a new `losses/special.py`:

```python
class SpecialBlockCountLoss(nn.Module):
    def __init__(self, block_size=16, output_stride=4): ...

    def forward(self, d_map, gt_y16, special_mask16):
        mu = sum_pool(d_map, self.block_size, self.output_stride).squeeze(1)
        y = gt_y16.to(mu.device).float()
        mask = special_mask16.to(mu.device).bool()
        if not mask.any():
            return d_map.new_zeros(())
        return F.smooth_l1_loss(mu[mask], y[mask])
```

## 12.11 Robustness

Keep hierarchical clean→degraded consistency:

\[
SmoothL1(\log(1+\mu^{deg}_B),\operatorname{stopgrad}\log(1+\mu^{clean}_B)).
\]

Photometric family should eventually cover brightness, contrast, gamma/exposure, color temperature, blur/motion blur, Gaussian/shot noise, salt-pepper, JPEG, optional haze.

---

# 13. Recommended initial loss weights

After 30% training:

| Term | Weight |
|---|---:|
| HNB | 1.00 |
| allocation | 0.50 |
| hard negative | 0.25 |
| empty | 0.50 |
| global log | 0.50 |
| direct count | 0.50 |
| special block | 0.25 |
| robustness | 0.10 |
| KD | 0.00 initially |

Do **not** enable KD until the non-KD student is stable.

---

# 14. Curriculum — implement as factors, never hard-coded replacement weights

Let the table below be multiplicative factors on configured lambdas.

| Progress | HNB | Alloc | HN | Empty | Global-log | Direct | Special | Robust | KD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–10% | 1.0 | 0.5 | 0 | 0 | 1.0 | 0.5 | 0 | 0 | 0 |
| 10–30% | 1.0 | 1.0 | 0.4 | 0.5 | 1.0 | 1.0 | 0.5 | 0 | 0.5 if enabled |
| 30–100% | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 if enabled |

Example:

```python
w_special = self.lambda_special * factor_special
```

Then `lambda_special=0` disables the term at all epochs.

---

# 15. Criterion API changes

## File

`losses/criterion.py`

Extend constructor:

```python
lambda_global: float = 0.5,
lambda_direct: float = 0.5,
lambda_special: float = 0.25,
lambda_kd: float = 0.0,
```

Add:

```python
self.direct_loss = GlobalCountLoss(mode="sqrt_normalized")
self.special_loss = SpecialBlockCountLoss(...)
```

Extend forward:

```python
def forward(
    self,
    d_map,
    gt_block_counts,
    gt_z_alloc,
    gt_counts,
    gt_special_mask16=None,
    d_degraded=None,
    teacher_map=None,
    progress=1.0,
):
```

Call allocation with special weights:

```python
weights16 = None
if gt_special_mask16 is not None:
    weights16 = 1.0 + gt_special_mask16.float()

l_alloc, alloc_details = self.alloc_loss(
    d_map,
    gt_z_alloc,
    y16,
    block_weights=weights16,
    return_details=True,
)
```

Log every component separately.

---

# 16. Optional knowledge distillation — training only

Do not make this necessary for the first SR48 result.

## 16.1 Why

Tiny students can gain accuracy from stronger training-time networks without increasing deployment size. This is especially relevant once SR48 is below 0.2M.

## 16.2 Recommended teacher-output distillation first

Avoid architecture-specific feature adapters in version 1.

Frozen teacher produces `D_T`; student produces `D_S`.

Hierarchical count-map distillation:

\[
L_{KD-count}=\frac1{|\mathcal B|}\sum_{B\in\mathcal B}
SmoothL1\left(
\log(1+\mu^{S}_B),
\operatorname{stopgrad}\log(1+\mu^{T}_B)
\right).
\]

Use the same hierarchy `{16,32,64}` or `{16,32,96}`.

Optionally distill within-block allocation only on GT-positive blocks:

\[
L_{KD-alloc}=\operatorname{mean}_{y_b>0}KL(p_b^T\|p_b^S).
\]

Total:

\[
L_{KD}=L_{KD-count}+0.25L_{KD-alloc}.
\]

Start:

```text
lambda_kd = 0.2
```

only after the non-KD student works.

## 16.3 Teacher choice

Use a teacher only if it materially outperforms the student baseline on validation.

Possible choices:

1. a larger 4-scale HPC teacher (64ch) trained with the same targets;
2. a strong public counting teacher if integration is clean;
3. the 0.35M current HPC-Lite only if its validation performance is better than the new student's early baseline.

Do not distill a teacher's known large-border failure blindly. Ground-truth special-block loss remains active.

---

# 17. Sampling

## 17.1 Preserve explicit empty group

Current training sampler must have a dedicated `C=0` group rather than letting quantile bins mix negatives with sparse positives.

For positive images:

\[
d_i=\log\left(1+10^6\frac{C_i}{H_iW_i}\right).
\]

Cross density bins with luminance bins; group weight:

\[
w_g\propto\frac1{\sqrt{n_g}}.
\]

## 17.2 Add scale-aware image/crop exposure

Image-level density sampling does not guarantee a dense or large-person crop. The transform mixture in Section 10 is therefore required even if the image sampler is balanced.

Do not build a second complicated sampler until the crop-level strategy is measured.

---

# 18. Dataset loaders

## 18.1 ShanghaiTech / QNRF / NWPU

Use **strict parsing by default**:

- missing annotation → raise for any train/val sample;
- malformed annotation → raise;
- test without public GT can explicitly set `has_gt=False`;
- never convert parser exceptions to empty point arrays.

## 18.2 NWPU

Keep official val/test split handling separate from diagnostics.

Do not infer a train/val split by scanning all images if the split file is missing; fail fast to avoid leakage.

## 18.3 Base dataset return schema

Recommended train sample:

```python
{
    "image": Tensor[3,H,W],
    "image_degraded": Tensor[3,H,W],
    "has_degraded": BoolTensor[],
    "gt_blocks": {B: Tensor[H/B,W/B]},
    "gt_z_alloc": Tensor[H/4,W/4],
    "gt_count": FloatTensor[],
    "gt_large_mask16": Tensor[H/16,W/16],
    "gt_true_border_mask16": Tensor[H/16,W/16],
    "gt_special_mask16": Tensor[H/16,W/16],
    "has_gt": BoolTensor[],
    "img_path": str,
}
```

All keys must be present for every training sample.

---

# 19. Evaluation metrics

## 19.1 Standard

\[
MAE=\frac1N\sum_i|\hat C_i-C_i|,
\]

\[
RMSE=\sqrt{\frac1N\sum_i(\hat C_i-C_i)^2}.
\]

## 19.2 NWPU official NAE

For `C_i>0` only:

\[
NAE=\operatorname{mean}\frac{|\hat C_i-C_i|}{C_i}.
\]

Zero-GT images are excluded from official NAE.

Do not use `(gt+1)` for the official metric.

## 19.3 Required new scale/border diagnostics

The new failure mode must be measured quantitatively, not only visually.

For each GT point in evaluation images compute:

\[
d_{border}=\min(x,y,W-1-x,H-1-y),
\]

and nearest-neighbor distance `d_nn`.

Recommended groups:

```text
true-border: d_border < 32
near-border: 32 <= d_border < 64
interior: d_border >= 64

dense/small proxy: d_nn < 16
medium: 16 <= d_nn < 48
large/isolated proxy: d_nn >= 48
```

For each point, measure predicted mass within a fixed local window, e.g. 32×32 or 64×64 input pixels, but report this only as a diagnostic.

Required metrics:

```text
point_local_mass_border_large
point_local_mass_interior_large
point_local_mass_border_dense
point_local_mass_interior_dense
miss_rate_border_large
```

Also retain:

- empty-image MAE;
- empty predicted mean/p95;
- empty 16×16 false-mass distribution;
- top 10% dense-image MAE;
- luminance diagnostics;
- official NWPU level/illumination groups when labels are available.

Do not call custom count bins "official NWPU groups".

---

# 20. Optimizer and training

## 20.1 Optimizer groups

Recommended:

```text
ShuffleNet pretrained backbone: 2.5e-5
new neck/router/head:         1.0e-4
criterion dispersion params: 1.0e-4
```

AdamW, `weight_decay=1e-4`.

Criterion parameters must be included:

```python
criterion = criterion.to(device)

optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": 2.5e-5},
    {"params": new_model_params, "lr": 1e-4},
    {"params": criterion.parameters(), "lr": 1e-4},
], weight_decay=1e-4)
```

Do not accidentally include the same model parameter in two groups.

## 20.2 Schedule

- 5% warmup;
- cosine decay;
- initial research run ~300–500 epochs;
- if SHA continues improving late, extend to 1000+ only after confirming no data bug/overfitting;
- effective batch ≈8 or larger if memory permits;
- AMP enabled;
- NB calculations float32;
- grad clip 1.0.

## 20.3 Training seeds

- 1 seed for smoke/debug;
- 3 seeds for development claims;
- 5 final seeds if compute budget permits.

---

# 21. Checkpointing

Checkpoint must contain:

```text
model_state_dict
criterion_state_dict
optimizer_state_dict
scheduler_state_dict
scaler_state_dict
epoch
global_step
best_metric
config
```

Optional exact-resume state:

```text
python RNG
numpy RNG
torch CPU RNG
torch CUDA RNG(s)
```

Never reset learned NB dispersions on resume.

---

# 22. File-by-file implementation plan

## `models/backbone.py`

**Keep:** `MobileNetV4Backbone`.  
**Add:** `ShuffleNetV2PyramidBackbone`.

Acceptance:

- outputs 4 features;
- exact 143,136 params;
- ImageNet weights load cleanly;
- no conv5/fc attached.

## `models/blocks.py`

**Keep:** existing conv/GN/DS blocks.  
**Add:**

- `MultiPoolContext`;
- `SimAM`;
- optional `make_group_norm(c)` helper.

Acceptance:

- `MultiPoolContext` and `SimAM` have zero trainable params;
- arbitrary spatial shape works.

## `models/neck.py`

**Keep:** `AdditiveFPNNeck`.  
**Add:** `ScaleRoutedFusionNeck`.

Acceptance:

- accepts channels `[24,48,96,192]`;
- output `/4`, 48ch;
- route weights sum to one along scale dim to numerical tolerance;
- odd input spatial dimensions do not crash.

## `models/hpc_lite.py`

**Keep:** `HPCLite`.  
**Add:** `HPCLiteSR48`.

Change new model prediction padding to multiple 32 and neutral normalized-zero padding.

Acceptance:

- deploy params ≈174,629;
- forward 448→112 map;
- forward 672→168 map;
- odd image inference uses ceil stride-4 dimensions;
- output finite and strictly positive.

## `datasets/transforms.py`

**Modify:** `GeometricTransforms` or add `ScaleAwareSafeGeometricTransforms`.

Must implement:

- isotropic scaling;
- `d_nn` calculation;
- candidate crop guard rejection;
- point-centered large/border sampling;
- true-border metadata;
- max attempts + deterministic fallback;
- horizontal flip preserving flags.

Strong recommendation: create a new transform class and keep the old one for ablation.

## `datasets/common.py`

**Modify:** train path to receive transform metadata and create fixed special masks.

Add returned keys:

```text
gt_large_mask16
gt_true_border_mask16
gt_special_mask16
```

Do not create optional dict keys.

## `targets/special_blocks.py` — new

Build fixed block masks from point flags.

Tests:

- exact block indexing at 0, 15.999, 16.0, right/bottom boundaries;
- shape correct for 448 and 672;
- flags survive flip.

## `losses/allocation.py`

**Modify:** optional `block_weights`, optional diagnostics.

Return details containing:

```text
alloc_ce
alloc_target_entropy
alloc_kl
```

Do not change the underlying target definition.

## `losses/hard_negative.py` or `losses/special.py`

**Add:** `SpecialBlockCountLoss`.

Ensure mask and target are moved to prediction device.

## `losses/criterion.py`

Add direct and special terms, optional KD term, and preserve ablation-safe curriculum.

## `losses/distillation.py` — optional new

Only after base SR48 is stable.

Implement hierarchical output-map distillation. Teacher must be frozen and run under `torch.no_grad()`.

## `metrics/scale_border.py` — new

Implement scale/border diagnostic grouping. This is evaluation-only.

## `metrics/counting.py`

Keep strict shape check and official NAE convention.

## `utils/checkpoint.py`

Keep criterion/scaler state handling.

## `utils/logging.py`

Use a fixed field schema or support safe schema expansion. Add router/alloc diagnostics.

---

# 23. Suggested config

```yaml
model:
  name: hpc_lite_sr48
  backbone: shufflenet_v2_x0_5
  pretrained: true
  neck_width: 48
  route_temperature: 1.0
  pool_kernels: [3, 5, 7]
  pool_residual_mix: 0.5
  simam_lambda: 1.0e-4
  head_kernel: 5
  output_stride: 4
  eps_d: 1.0e-6

loss:
  lambda_hnb: 1.0
  lambda_alloc: 0.5
  lambda_hn: 0.25
  lambda_empty: 0.5
  lambda_global: 0.5
  lambda_direct: 0.5
  lambda_special: 0.25
  lambda_rob: 0.1
  lambda_kd: 0.0
  hard_negative_fraction: 0.10
  special_alloc_beta: 1.0
  use_stratified_nb: true

augmentation:
  scale_range: [0.75, 2.0]
  flip_prob: 0.5
  safe_crop: true
  max_crop_attempts: 20
  crop_guard_nn_factor: 0.20
  crop_guard_min_px: 8
  crop_guard_max_px: 48
  large_nn_threshold_px: 48
  true_border_threshold_px: 32
  random_crop_prob: 0.75
  large_center_crop_prob: 0.15
  border_center_crop_prob: 0.10
  degraded_prob: 0.30

train:
  optimizer: adamw
  backbone_lr: 2.5e-5
  new_layers_lr: 1.0e-4
  criterion_lr: 1.0e-4
  weight_decay: 1.0e-4
  warmup_fraction: 0.05
  schedule: cosine
  grad_clip: 1.0
  amp: true
```

Dataset overrides:

```yaml
SHA:
  crop_size: 448
  hnb_blocks: [16, 32, 64]

SHB:
  crop_size: 448
  hnb_blocks: [16, 32, 64]

QNRF:
  crop_size: 672
  hnb_blocks: [16, 32, 96]

NWPU:
  crop_size: 672
  hnb_blocks: [16, 32, 96]
```

---

# 24. Regression tests required before real training

Create `tests/test_sr48.py` and retain existing target/loss tests.

## 24.1 Architecture

```text
[ ] backbone param count = 143136
[ ] full deploy param count ~174629
[ ] no classifier/conv5 in backbone
[ ] route weights sum to 1
[ ] SimAM params = 0
[ ] MultiPoolContext params = 0
[ ] output positive and finite
```

## 24.2 Shapes

```text
[ ] 448×448 -> 112×112
[ ] 672×672 -> 168×168
[ ] 449×451 predict -> ceil(449/4) × ceil(451/4)
```

## 24.3 Targets

```text
[ ] allocation global mass = N
[ ] allocation per-16-block mass = exact y16
[ ] block count sums at all scales = N
[ ] special mask boundary indexing correct
```

Stress test at ~20k points.

## 24.4 Crop safety

Construct synthetic image with annotated center just outside a candidate crop but inside guard region.

```text
[ ] unsafe candidate rejected
[ ] true image edge itself is not treated as synthetic negative region
[ ] large-centered crop retains selected point
[ ] flip maintains point/flag association
```

## 24.5 Criterion

```text
[ ] forward/backward finite under AMP
[ ] NB path float32
[ ] all lambdas=0 -> total loss exactly 0 at progress 0.05, 0.20, 0.50
[ ] special loss returns zero if no special mask
[ ] criterion parameters receive gradients and are optimizer parameters
[ ] allocation KL >= approximately 0 (allow tiny numerical tolerance)
```

## 24.6 Dataset integrity

```text
[ ] missing SHA/QNRF/NWPU annotation raises
[ ] malformed point array raises
[ ] NWPU test can explicitly contain has_gt=False without being treated as zero GT
[ ] train batch default_collate works for mixed degraded flags
```

## 24.7 Metrics

```text
[ ] unequal prediction/GT lengths raise
[ ] NWPU official NAE excludes gt=0
[ ] empty diagnostics separate from NAE
```

---

# 25. Ablation plan — do not change multiple ideas blindly

Run in this order.

### A0 — current baseline

`HPCLite`, ~0.350M.

Record current MAE and border/scale diagnostics.

### A1 — safe crop only

Current architecture, new crop logic.

Purpose: estimate how much of large-border failure is a data-label artifact.

### A2 — SR48 architecture only

New backbone + scale router, old loss/data except already mandatory correctness fixes.

Purpose: isolate architecture.

### A3 — SR48 + safe scale-aware crop

Expected main model.

### A4 — + direct count loss

Measure MAE/count conservation improvement.

### A5 — + special-block weighting/loss

Specifically measure border-large diagnostic improvement.

### A6 — + SimAM ablation

Compare with `nn.Identity()`.

### A7 — + multi-pool context ablation

Compare no pool, `{3}`, `{3,5,7}`.

### A8 — KD

Only after A3–A7 stable.

This order is important for a defensible paper. Do not report one giant architecture change without component evidence.

---

# 26. Acceptance gates before a 300+ epoch run

Do not start expensive training unless all are true:

1. all unit tests pass;
2. deploy params are profiled;
3. 448/672/odd inference is correct;
4. a 100-batch dataloader stress test has no schema/annotation failure;
5. a 1–2 epoch smoke run has finite loss/gradients;
6. route usage is non-NaN and not trivially broken;
7. special-mask rate is logged (not always zero / always one);
8. predicted count changes in the correct direction on a tiny overfit subset.

### Tiny overfit test

Take 4–8 training crops and train until strongly overfit.

Expected:

- global/direct count losses become small;
- HNB decreases toward its non-zero likelihood floor;
- allocation raw CE may remain >0 due target entropy;
- allocation **excess KL** should approach zero;
- MAE on the tiny set should approach zero or very small values.

If tiny-set MAE cannot overfit, do not launch the full run.

---

# 27. Logging schema

At minimum log:

```text
epoch
step
lr_backbone
lr_new
loss_total
loss_hnb
loss_alloc_ce
alloc_target_entropy
alloc_kl
loss_hn
loss_empty
loss_global
loss_direct
loss_special
loss_rob
loss_kd
r16
r32
r64_or_r96
route_mean_s4
route_mean_s8
route_mean_s16
route_mean_s32
route_entropy
train_mae
val_mae
val_rmse
empty_mae
border_large_local_mass
border_large_miss_rate
```

Use a stable schema from run start.

---

# 28. What must remain training-only

The following must **not** appear in deployment parameter count or inference graph:

- HNB dispersion parameters;
- target generators;
- special scale/border masks;
- density/luminance sampler logic;
- degraded-view teacher consistency;
- KD teacher and KD adapters;
- route diagnostics;
- all GT point processing.

Deploy graph ends at `D` and `sum(D)`.

---

# 29. What not to add now

To preserve the ultra-lightweight research story, do **not** add by default:

- Mamba decoder;
- transformer decoder;
- Hungarian matching;
- occupancy/detection inference head;
- full RCMB stack copied from LRMBNet;
- concatenative 64/128-channel decoder;
- large QKV attention windows;
- Gaussian density-map target as a replacement for HPC allocation target;
- PML/ZIP into the core model.

PML/ZIP can remain external baselines/ablations if needed.

---

# 30. Expected failure modes and contingency actions

## 30.1 Router collapses to `/4`

Symptoms:

```text
route_mean_s4 > 0.9
others ~0
```

Actions, in order:

1. verify normalized feature scales and GN;
2. lower router LR if unstable;
3. initialize router weights to zero so initial softmax is uniform;
4. optionally add `lambda_route_balance=0.01`.

Recommended initialization:

```python
nn.init.zeros_(self.router.weight)
nn.init.zeros_(self.router.bias)
```

This makes the initial routing exactly uniform.

## 30.2 Large-border cases still missed

Before adding params:

1. inspect safe-crop rejection rate;
2. verify true-border crops are actually sampled;
3. inspect `gt_special_mask16` occupancy;
4. increase `lambda_special` 0.25→0.5;
5. increase special allocation beta 1→2;
6. only then consider replacing DW5×5 with DW7×7.

DW7×7 at 48 channels adds only:

\[
48(49-25)=1,152
\]

weights, so total remains around 0.176M.

## 30.3 Small/distant accuracy gets worse

- inspect router: `/4` and `/8` should remain active in dense regions;
- lower special loss if it distorts global training;
- compare 48ch vs 40ch only after verifying architecture, not before;
- do not remove `/4` high-resolution feature.

## 30.4 MAE good but FLOPs too high

- route at `/16` instead of `/8` as an ablation;
- remove SimAM before shrinking channels;
- use single 3×3 pool context instead of `{3,5,7}`;
- profile interpolation overhead on deployment hardware.

---

# 31. Paper positioning if successful

The clean story is:

> A sub-0.2M crowd counter that combines a scale-routed four-resolution student with hierarchical probabilistic count supervision. The model uses parameter-free contextual modulation and training-only scale/border-aware supervision to improve perspective and edge robustness without increasing deployment complexity.

Potential core contributions, **only if supported by ablations**:

1. **Scale-Routed Ultra-Light Student:** location-dependent selection across `/4,/8,/16,/32` at ~0.175M parameters.
2. **Hierarchical Probabilistic Supervision:** exact multi-scale NB block-count modeling + block-constrained allocation.
3. **Border/Scale-Safe Training:** prevents synthetic crop boundaries from creating false-negative large-person examples and explicitly reweights rare near-camera/true-border regions.
4. **Parameter-Free Context Modulation:** coarse multi-pool context + SimAM without deploy parameter increase.

Do not claim novelty simply because SimAM or pooling is used; those are established modules. Novelty must come from the integrated counting formulation and validated scale/border mechanism.

---

# 32. Research references for implementation rationale

1. **LRMBNet** — *Lightweight Res-Connection Multi-Branch Network for Highly Accurate Crowd Counting and Localization.* Reported V0.5: 0.25M params, SHA MAE 59.94, QNRF MAE 93.90.  
   https://www.sciencedirect.com/org/science/article/pii/S1546221824002893

2. **Context-Aware Crowd Counting**, CVPR 2019 — adaptive importance of multiple receptive-field sizes per spatial location.  
   https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Context-Aware_Crowd_Counting_CVPR_2019_paper.html

3. **STEERER**, ICCV 2023 — selective inheritance for scale variation; official code available.  
   https://openaccess.thecvf.com/content/ICCV2023/html/Han_STEERER_Resolving_Scale_Variations_for_Counting_and_Localization_via_Selective_ICCV_2023_paper.html  
   https://github.com/taohan10200/STEERER

4. **SimAM**, ICML 2021 — parameter-free attention.  
   https://proceedings.mlr.press/v139/yang21o

5. **PoolFormer / MetaFormer**, CVPR 2022 — simple non-parametric pooling as token mixer.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Yu_MetaFormer_Is_Actually_What_You_Need_for_Vision_CVPR_2022_paper.html

6. **HybridCount**, Pattern Recognition 2026 — scale-aware knowledge distillation; HybridCount-S reported at 0.22M params and 52.25 MAE on SHT-A.  
   https://www.sciencedirect.com/science/article/pii/S0031320326000063

7. **ReviewKD for Crowd Counting** — training-time knowledge transfer for lightweight counters.  
   https://arxiv.org/abs/2206.05475

8. **PyTorch ShuffleNetV2 implementation** — source architecture used for the exact backbone parameter calculation.  
   https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py

---

# 33. Implementation order for the next agent

Follow this exact sequence to minimize debugging ambiguity.

### Stage 1 — establish clean baseline

1. Start from previously fixed files / apply bugfix patch.
2. Run existing smoke tests.
3. Save current 0.350M baseline profile and validation outputs.

### Stage 2 — architecture only

4. Add `ShuffleNetV2PyramidBackbone`.
5. Add `MultiPoolContext` and `SimAM`.
6. Add `ScaleRoutedFusionNeck`.
7. Add `HPCLiteSR48`.
8. Run shape/parameter/gradient tests.
9. Profile exact params/MACs/FPS.

### Stage 3 — data fix

10. Add safe scale-aware crop transform.
11. Add special-block target builder.
12. Return fixed masks from dataset.
13. Stress-test DataLoader.

### Stage 4 — loss integration

14. Extend allocation loss with optional weights + entropy/KL diagnostics.
15. Add direct count loss.
16. Add special-block count loss.
17. Extend criterion and curriculum.
18. Confirm all-zero-lambda ablation test.

### Stage 5 — evaluation

19. Add scale/border diagnostics.
20. Run tiny overfit.
21. Run 1–2 epoch smoke.
22. Only then start full training.

### Stage 6 — optional KD

23. Implement hierarchical output KD only after base SR48 is validated.

---

# 34. Deliverables expected from the coding agent

The next coding agent should return:

```text
1. patched source tree / zip
2. unified diff patch
3. updated config(s)
4. test_sr48.py + existing regression tests
5. parameter profile report
6. shape profile for 448/672/odd input
7. 1–2 epoch smoke log
8. list of any deviations from this spec and reasons
```

The agent must not silently alter the mathematics, block scales, target semantics, or evaluation protocol.

---

# 35. Copy-paste task prompt for the next coding agent

```text
Implement HPC-Lite-SR48 exactly according to HPC_Lite_SR48_Final_Implementation_Proposal.md.

Start from the previously bug-fixed HPC-Lite source, not the unaudited raw files. Keep the old HPCLite class as a baseline and add HPCLiteSR48 separately.

Priority order:
1) ShuffleNetV2 x0.5 feature backbone through /32;
2) 48-channel four-scale independent refinements;
3) /8 local 4-way scale router, weighted fusion at /4;
4) zero-parameter multi-pool context at /32 and SimAM after fusion;
5) DW5x5 Softplus stride-4 count-mass head;
6) safe scale-aware crop that prevents artificial-border false negatives;
7) fixed 16x16 large/true-border/special block masks;
8) direct count + special-block loss and special-weighted allocation;
9) allocation entropy/KL diagnostics;
10) full regression tests and exact param profiling.

Do not add Mamba, Transformer/QKV attention, matching, detection heads, PML/ZIP core losses, or full LRMBNet RCMB blocks.

Target deploy parameters: approximately 174,629 (<0.20M). Verify this with runtime parameter counting. Preserve exact count-mass semantics and HNB/allocation targets.

Before long training, prove:
- tests pass;
- 448 -> 112, 672 -> 168;
- odd-size predict uses ceil/4;
- all-zero lambdas produce zero total loss at every curriculum phase;
- target mass conservation holds;
- missing annotations fail fast;
- DataLoader schema is stable;
- tiny subset can overfit in count MAE;
- route weights are finite and sum to one.

Return a unified patch, complete fixed source zip, tests, and a short implementation report.
```

---

# 36. Final decision summary

The recommended production research path is **not** to grow the current 0.35M MobileNetV4 model.

The final proposed student is:

\[
\boxed{
\text{ShuffleNetV2-0.5 pyramid}
+\text{48ch scale-routed fusion}
+\text{0-param context/attention}
+\text{HPC probabilistic supervision}
}
\]

with estimated:

\[
\boxed{0.175M\ \text{deploy params}}
\]

and with the training pipeline explicitly corrected for the discovered large-person / true-border failure.

The decisive experimental question is not whether the parameter count can be reduced—it can—but whether **scale routing + safe border training** moves the model onto a better accuracy/parameter Pareto frontier than LRMBNet V0.5 and the current HPC-Lite baseline.
