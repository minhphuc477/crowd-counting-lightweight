# HPC-Lite: Final Research & Engineering Specification

**Working title:** *HPC-Lite: Hierarchical Probabilistic Crowd Counting for Robust Lightweight Deployment*  
**Document purpose:** final handoff specification for implementation/research agents.  
**Status:** implementation-ready research specification; reported accuracy/parameter targets are goals, **not experimental claims**.  
**Date:** 2026-08-25

---

## 0. Executive decision

The project is no longer defined as “PML + ZIP”. The final core method is a **single-map lightweight crowd counter** trained with several training-only objectives designed around the hardest failure modes of real crowd-counting datasets, especially NWPU-Crowd:

1. **hard negative / empty scenes** — suppress hallucinated people explicitly;
2. **extreme density** — supervise counts hierarchically at several spatial scales;
3. **large count overdispersion** — use a Negative-Binomial (NB) likelihood rather than assuming Poisson variance;
4. **local spatial structure** — supervise how count mass is distributed inside small blocks without Gaussian density-map targets;
5. **dark / blur / noisy conditions** — use photometric degradation training and count-consistency with no inference-time module;
6. **lightweight deployment** — inference contains only a small backbone, a 32-channel additive neck, and a one-channel positive mass map.

The deployed model is intentionally simple:

```text
Image
  ↓
MobileNetV4-Conv-Small-0.5 trunk
  ↓
features @ 1/4, 1/8, 1/16
  ↓
32-channel additive depthwise-separable FPN
  ↓
1-channel Softplus count-mass map D @ stride 4
  ↓
sum(D)
  ↓
predicted crowd count
```

All hierarchical likelihoods, hard-negative mining, degradation consistency, optional PML/ZIP/OT/KD branches, and dataset balancing are **training-only**.

The main design principle is:

> **Spend deployment capacity on representation, and use training-time probabilistic, spatial, robustness, and hard-negative supervision to teach that limited capacity efficiently.**

---

# 1. Research objective

Design a crowd-counting model that is simultaneously:

- lightweight enough for practical deployment;
- accurate across sparse and dense datasets;
- robust to NWPU-style negative scenes;
- robust to extremely dense images;
- less sensitive to low illumination, blur, compression, and noise;
- trained directly from point annotations without requiring Gaussian density maps;
- simple at inference.

The primary datasets are:

- ShanghaiTech Part A (SHA)
- ShanghaiTech Part B (SHB)
- UCF-QNRF
- NWPU-Crowd

Optional later validation:

- JHU-Crowd++

---

# 2. Core hypotheses

## H1 — hierarchical count supervision helps small models

A small model may not resolve every individual head in an extremely dense patch, but it can still estimate mass at larger spatial supports. Therefore the same output map should receive count supervision at several block sizes.

## H2 — Negative Binomial is a better default than Poisson for heterogeneous crowd blocks

Crowd block counts can be overdispersed:

\[
\operatorname{Var}(Y) > \mathbb E[Y].
\]

Poisson requires:

\[
\operatorname{Var}(Y)=\mu.
\]

Negative Binomial allows:

\[
\operatorname{Var}(Y)=\mu+\frac{\mu^2}{r}.
\]

## H3 — explicit false-positive suppression is necessary

On empty and background-heavy scenes, average density/count losses may not focus sufficiently on the few background regions that hallucinate the most crowd mass. Hard-zero blocks therefore require targeted mining.

## H4 — fine spatial supervision should not scale linearly with number of people

A dense block containing 300 annotations should not create 300 times more localization gradient than a block containing one annotation. Local allocation supervision therefore normalizes by local count.

## H5 — robustness should be mostly a training problem

Low-light enhancement networks, extra inference branches, and heavy domain modules violate the deployment goal. Robustness should first be pursued with photometric degradation, clean→degraded consistency, balanced sampling, and optional distillation.

---

# 3. Non-goals and scope control

Do **not** add the following to the main model unless a controlled ablation later proves a clear benefit:

- Transformer decoder;
- Mamba decoder;
- DETR/P2P fixed queries;
- Hungarian matching in the core method;
- a second deployment head;
- image-enhancement network at inference;
- multi-scale test-time ensemble in the main efficiency table;
- PCGrad / gradient surgery in the core method;
- PML or ZIP as required main losses;
- knowledge distillation as a required main component.

These are optional research branches only.

---

# 4. Final inference architecture

## 4.1 Backbone

Primary backbone:

```text
mobilenetv4_conv_small_050
```

Use ImageNet pretrained weights if available.

The current timm model card reports **2.2M parameters for the complete classification model**. The project does **not** use the classifier and should ideally physically truncate the network after the feature stage required for reduction 16. The exact final parameter count must be obtained from the implemented model; do not claim 1.x M until measured.

Reference:

- timm model card: https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
- timm: https://github.com/huggingface/pytorch-image-models
- MobileNetV4 paper: https://arxiv.org/abs/2404.10518

### Required backbone outputs

Select feature maps with effective reductions:

\[
C_4,\; C_8,\; C_{16}
\]

where spatial resolutions are approximately:

\[
H/4\times W/4,
\quad H/8\times W/8,
\quad H/16\times W/16.
\]

**Implementation rule:** select features by `feature_info.reduction()` or an equivalent programmatic check. Do not hard-code stage indices without verifying shapes.

### Prototype versus final deploy implementation

Prototype may use:

```python
features_only=True
```

for correctness.

However, `out_indices`/feature wrappers do not automatically prove unused later layers are absent from the parameter count. Before publishing efficiency results, inspect `named_parameters()` and create a physically truncated backbone if necessary.

---

## 4.2 Neck width

Core neck width:

\[
C=32.
\]

Ablation widths:

\[
C\in\{24,32,48\}.
\]

Do not start with 64 channels; lightweight baselines already make a 2–3M model insufficiently distinctive.

---

## 4.3 Lateral projections

For each selected feature:

\[
L_s=\phi\left(N\left(\operatorname{Conv}_{1\times1}(C_s)\right)\right),
\]

where:

- output channels = 32;
- \(\phi\) = SiLU;
- recommended neck normalization = `GroupNorm(8, 32)` for small-batch stability.

Keep the backbone's native pretrained normalization. If batch size is small, compare:

1. train backbone BN normally;
2. freeze BN running statistics but retain affine parameters.

Do not silently replace all MobileNet normalization without an ablation.

---

## 4.4 Cheap coarse context block

At reduction 16:

\[
Q_d=DWConv_{3\times3,d}(L_{16}),
\quad d\in\{1,2,3\}.
\]

Then:

\[
P_{16}=R\left(L_{16}+Q_1+Q_2+Q_3\right).
\]

This gives larger receptive field at very low parameter cost.

### DS residual refinement

Define:

\[
R(x)=\phi\left(x+PW_{1\times1}\left(\phi(N(DW_{3\times3}(x)))\right)\right).
\]

A minimal PyTorch form:

```python
class DSResidual(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.n1 = nn.GroupNorm(8, c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.n2 = nn.GroupNorm(8, c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.act(self.n1(self.dw(x)))
        y = self.n2(self.pw(y))
        return self.act(x + y)
```

This design is motivated by the practical lightweight refinement pattern used in the ZIP repository (`DW 3×3 → PW 1×1 → residual`):

- https://github.com/Yiming-M/ZIP/blob/main/models/utils/refine.py

---

## 4.5 Additive top-down fusion

Use bilinear interpolation and addition:

\[
P_8=R\left(L_8+\operatorname{Up}(P_{16})\right),
\]

\[
P_4=R\left(L_4+\operatorname{Up}(P_8)\right).
\]

Use:

```python
F.interpolate(..., mode="bilinear", align_corners=False)
```

Do **not** concatenate by default; concatenation increases channels, memory traffic, and subsequent convolution cost.

---

## 4.6 One-channel mass head

Head:

\[
H=\phi\left(N(DWConv_{3\times3}(P_4))\right)
\]

\[
z=Conv_{1\times1}(H)
\]

\[
\boxed{D=\operatorname{softplus}(z)+\epsilon_D}
\]

with:

\[
\epsilon_D=10^{-6}.
\]

Interpretation:

> \(D_{ij}\) is predicted **count mass for one stride-4 cell**, not a density value that must later be multiplied by pixel area.

Image count:

\[
\boxed{\hat C=\sum_{i,j}D_{ij}}.
\]

### Critical initialization rule

Do **not** initialize the final head with zero bias and Softplus. Since:

\[
\operatorname{softplus}(0)\approx0.693,
\]

a large output grid would initially predict thousands of people.

Preferred data-driven bias:

1. compute mean training crop count \(\bar C\);
2. compute number of output cells \(M\);
3. set initial cell mass:

\[
m_0=\max(\bar C/M,10^{-5});
\]

4. initialize final bias with the inverse Softplus:

\[
b_0=\log(e^{m_0}-1).
\]

Stable implementation:

```python
def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))
```

Fallback if training statistics are not ready:

```text
head bias = -6
```

but this fallback must not replace the measured initialization in final experiments.

---

# 5. Reference model skeleton

```python
class HPCLite(nn.Module):
    def __init__(self, backbone, in_channels, width=32):
        super().__init__()
        self.backbone = backbone

        c4, c8, c16 = in_channels

        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        self.ctx1 = DepthwiseDilated(width, dilation=1)
        self.ctx2 = DepthwiseDilated(width, dilation=2)
        self.ctx3 = DepthwiseDilated(width, dilation=3)

        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

        self.head_dw = nn.Conv2d(
            width, width, 3, padding=1,
            groups=width, bias=False
        )
        self.head_norm = nn.GroupNorm(8, width)
        self.head_act = nn.SiLU(inplace=True)
        self.head_out = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        c4, c8, c16 = self.backbone(x)

        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        p16 = self.ref16(
            l16
            + self.ctx1(l16)
            + self.ctx2(l16)
            + self.ctx3(l16)
        )

        p8 = self.ref8(
            l8 + F.interpolate(
                p16,
                size=l8.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        p4 = self.ref4(
            l4 + F.interpolate(
                p8,
                size=l4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        z = self.head_out(
            self.head_act(
                self.head_norm(self.head_dw(p4))
            )
        )

        return F.softplus(z) + 1e-6
```

**Acceptance condition:** output spatial reduction must be exactly 4 for every supported training crop and for arbitrary validation dimensions after padding/cropping.

---

# 6. Target construction: separate exact count targets from soft spatial targets

This distinction is mandatory.

Do **not** obtain NB block counts by sum-pooling a bilinearly splatted soft point map. Bilinear splatting near block boundaries can leak fractional mass into adjacent blocks, whereas Negative Binomial count targets must represent the actual integer number of annotated people in each block.

Use two target families.

---

## 6.1 Exact integer block counts for probabilistic counting

For each GT point \((x_n,y_n)\) and block size \(B\):

\[
b_x=\left\lfloor\frac{x_n}{B}\right\rfloor,
\qquad
b_y=\left\lfloor\frac{y_n}{B}\right\rfloor.
\]

Increment:

\[
Y^{(B)}[b_y,b_x]\mathrel{+}=1.
\]

Therefore:

\[
Y^{(B)}\in\mathbb N_0^{H_B\times W_B}.
\]

For a crop that is exactly divisible by \(B\), count conservation must satisfy:

\[
\sum_bY_b^{(B)}=N.
\]

These integer maps are used for:

- Negative-Binomial NLL;
- zero-block identification;
- hard-negative mining;
- density regime statistics.

---

## 6.2 Block-constrained soft allocation target

Fine allocation uses a base block \(B_A\), default:

\[
B_A=16\text{ pixels}.
\]

Output stride:

\[
s_D=4,
\]

so a 16-pixel allocation block contains:

\[
K=(16/4)^2=16
\]

output cells.

For each point:

1. assign it to its exact 16×16 input block;
2. convert position to stride-4 cell-center coordinates **inside that block**;
3. bilinearly splat only into cells belonging to that same block;
4. if a neighbor would fall outside the block, remove that neighbor and renormalize remaining weights;
5. ensure the weights for each point sum to exactly 1.

This creates soft block target \(Z_b\) with:

\[
\sum_kZ_{bk}=Y_b^{(16)}.
\]

This target is used only for spatial allocation.

### Coordinate convention

Stride-4 cell centers occur at:

\[
(2+4i,\;2+4j).
\]

For a point in input coordinates:

\[
u=x/4-1/2,
\qquad
v=y/4-1/2.
\]

Border handling must renormalize valid bilinear weights to preserve mass.

---

# 7. Hierarchical probabilistic count supervision

The same map \(D\) is supervised at several block scales.

For a block \(b\) at input block size \(B\):

\[
\mu_b^{(B)}=\sum_{(i,j)\in\Omega_b^{(B)}}D_{ij}.
\]

Since \(B\) is divisible by output stride 4, this is sum-pooling with kernel/stride:

\[
k=B/4.
\]

No separate count head is needed.

---

## 7.1 Dataset-specific block scales

Training crops should be exactly divisible by every configured probabilistic block size.

Recommended initial configuration:

| Dataset | Crop | Output grid | Hierarchical block sizes |
|---|---:|---:|---|
| SHA | 448 | 112×112 | 16, 32, 64 |
| SHB | 448 | 112×112 | 16, 32, 64 |
| UCF-QNRF | 672 | 168×168 | 16, 32, 96 |
| NWPU | 672 | 168×168 | 16, 32, 96 |

Why 96 rather than 64 for 672 crops:

\[
672/96=7
\]

is integral, whereas:

\[
672/64=10.5.
\]

This avoids silently dropping boundary regions during pooling.

**Do not use `avg_pool2d` and multiply by area. Use exact sum pooling.**

Reference implementation:

```python
def sum_pool(x, input_block_size: int, output_stride: int = 4):
    assert input_block_size % output_stride == 0
    k = input_block_size // output_stride
    assert x.shape[-2] % k == 0
    assert x.shape[-1] % k == 0
    return F.avg_pool2d(x, kernel_size=k, stride=k) * (k * k)
```

---

# 8. Negative-Binomial likelihood

Parameterization:

\[
Y\sim NB(\mu,r)
\]

with mean:

\[
\mathbb E[Y]=\mu
\]

and variance:

\[
\operatorname{Var}(Y)=\mu+\frac{\mu^2}{r}.
\]

Probability mass:

\[
P(Y=y)=
\frac{\Gamma(y+r)}{\Gamma(r)\Gamma(y+1)}
\left(\frac{r}{r+\mu}\right)^r
\left(\frac{\mu}{r+\mu}\right)^y.
\]

NLL:

\[
\begin{aligned}
\ell_{NB}(y,\mu,r)=-\bigg[
&\log\Gamma(y+r)-\log\Gamma(r)-\log\Gamma(y+1)\\
&+r(\log r-\log(r+\mu))\\
&+y(\log(\mu+\epsilon)-\log(r+\mu))
\bigg].
\end{aligned}
\]

---

## 8.1 Dispersion parameter

Use one learnable positive scalar per block scale:

\[
r_B=\operatorname{softplus}(\eta_B)+10^{-4}.
\]

This adds effectively no inference cost; these scalars are training criterion parameters and can be discarded after training.

### Recommended initialization by method of moments

Using **training data only**, calculate block-count mean and variance at each scale:

\[
m_B=\mathbb E[Y_B],
\qquad
v_B=\operatorname{Var}(Y_B).
\]

If:

\[
v_B>m_B,
\]
initialize:

\[
r_B^{(0)}=\frac{m_B^2}{v_B-m_B}.
\]

Otherwise initialize a large value such as:

\[
r_B^{(0)}=100,
\]

which approximates a less-overdispersed regime.

Convert to raw parameter using inverse Softplus.

---

## 8.2 Stable PyTorch implementation

NB calculations must run in float32 even under AMP.

```python
def nb_nll(y, mu, r, eps=1e-8):
    # y: non-negative integer-valued float tensor
    # mu: positive predicted mean
    # r: positive scalar/tensor broadcastable to mu
    y = y.float()
    mu = mu.float().clamp_min(eps)
    r = r.float().clamp_min(eps)

    log_r_plus_mu = torch.log(r + mu)

    log_prob = (
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )

    return -log_prob
```

Mandatory stress tests:

- `y=0, mu≈1e-6`;
- `y=0, mu=100`;
- `y=1`;
- `y=1000`;
- `y=20000`;
- backward under AMP training loop.

No NaN/Inf is acceptable.

---

# 9. Density-stratified block risk

Empty blocks can dominate the total number of blocks, while extreme dense blocks are rare but important. Therefore do not simply mean-reduce NB NLL across every block.

For each scale, compute positive training block-count quantiles using training data only:

\[
q_{50,B}^{+},\qquad q_{90,B}^{+}.
\]

Define groups:

\[
G_0: y=0
\]

\[
G_1: 0<y\le q_{50,B}^{+}
\]

\[
G_2: q_{50,B}^{+}<y\le q_{90,B}^{+}
\]

\[
G_3: y>q_{90,B}^{+}.
\]

Within each minibatch:

\[
R_g=\operatorname{mean}_{b\in G_g}\ell_b.
\]

Average only groups present in the minibatch:

\[
\boxed{
L_{NB}^{(B)}=
\frac{1}{|\mathcal G_{present}|}
\sum_{g\in\mathcal G_{present}}R_g
}
\]

and:

\[
\boxed{
L_{HNB}=\frac{1}{|\mathcal S|}\sum_{B\in\mathcal S}L_{NB}^{(B)}.
}
\]

This is a practical group-balanced risk strategy, not a claim of inventing group DRO.

Ablate against ordinary mean NB NLL.

---

# 10. Local spatial allocation loss

Within each allocation block \(b\):

\[
\mu_b=\sum_kD_{bk}.
\]

Normalize predicted cell masses:

\[
p_{bk}=\frac{D_{bk}+\epsilon}{\mu_b+K\epsilon}.
\]

Let block-constrained soft target be \(Z_{bk}\), satisfying:

\[
\sum_kZ_{bk}=y_b.
\]

For positive blocks:

\[
\boxed{
\ell_{alloc,b}
=-\frac{1}{y_b}
\sum_kZ_{bk}\log(p_{bk}+\epsilon)
}
\]

and:

\[
\boxed{
L_{alloc}=
\frac1{|\mathcal B_+|}
\sum_{b:y_b>0}\ell_{alloc,b}.
}
\]

The \(1/y_b\) normalization makes every positive block approximately one spatial-supervision unit regardless of whether it contains one or hundreds of points.

If there are no positive blocks in a batch:

```python
loss_alloc = D.new_zeros(())
```

This objective is closer to a normalized local allocation cross-entropy than a strict integer Multinomial likelihood because the target uses fractional bilinear mass. Describe it accurately in the paper.

---

# 11. Hard-negative false-mass mining

SCALNet's code contains an explicit false-positive loss that selects background predictions above a threshold. The new method keeps the idea of targeted false-positive suppression but removes the fixed probability threshold and uses predicted crowd mass itself.

Reference:

- https://github.com/WangyiNTU/SCALNet/blob/main/src/loc_loss.py

At base block size 16, identify exact zero blocks:

\[
\mathcal Z=\{b:Y_b^{(16)}=0\}.
\]

Hardness:

\[
h_b=\mu_b^{(16)}.
\]

For each image independently, select top fraction:

\[
k=\max\left(1,\lfloor\rho|\mathcal Z|\rfloor\right)
\]

with initial:

\[
\rho=0.10.
\]

Then:

\[
\boxed{
L_{HN}=\operatorname{mean}_{b\in TopK(\mathcal Z)}
\operatorname{SmoothL1}(\mu_b,0).
}
\]

Use per-image mining so a single image cannot monopolize all hard zero blocks in the batch.

Reference code:

```python
def hard_negative_mass_loss(pred_blocks, gt_blocks, top_fraction=0.10):
    losses = []
    for pred_i, gt_i in zip(pred_blocks, gt_blocks):
        zero = gt_i.eq(0)
        vals = pred_i[zero]
        if vals.numel() == 0:
            continue
        k = max(1, int(math.ceil(top_fraction * vals.numel())))
        hard = vals.topk(k, largest=True).values
        losses.append(F.smooth_l1_loss(hard, torch.zeros_like(hard)))

    if not losses:
        return pred_blocks.new_zeros(())
    return torch.stack(losses).mean()
```

Ablate \(\rho\in\{0.05,0.10,0.20\}\).

---

# 12. Whole-image empty suppression

Bayesian Crowd Counting's released loss explicitly handles images with no points by comparing total predicted density mass to zero.

Reference:

- https://github.com/zhiheng-ma/Bayesian-Crowd-Counting/blob/master/losses/bay_loss.py

For images where:

\[
C=0,
\]

the core empty-image objective is:

\[
L_{empty}=\operatorname{mean}\hat C.
\]

Because Softplus head initialization is controlled, this is safe after proper initialization.

Optional stabilization during the first few epochs:

\[
L_{empty}^{warm}=\log(1+\hat C),
\]

then switch to linear count mass after warm-up.

Do not use an occupancy classifier in the main method unless experiments show this direct mass objective is inadequate.

---

# 13. Global count objective

Hierarchical NB provides raw local count supervision, while the image-level count term helps long-range conservation.

Core initial form:

\[
\boxed{
L_{global}=
\operatorname{SmoothL1}
\left(
\log(1+\hat C),
\log(1+C)
\right).
}
\]

Ablations:

1. raw L1 count;
2. log SmoothL1;
3. square-root normalized residual:

\[
\operatorname{SmoothL1}
\left(
\frac{\hat C-C}{\sqrt{C+1}},0
\right).
\]

Do not invent a superior count loss before ablation evidence.

---

# 14. Adverse-condition robust training

The deploy model must handle low illumination, blur, compression, and sensor-like noise without adding an enhancement network.

The released ZIP NWPU configuration already includes brightness, contrast, saturation, blur, and salt/pepper perturbations:

- https://github.com/Yiming-M/ZIP/blob/main/configs/nwpu.yaml

Core robustness policy extends this conservatively.

---

## 14.1 Photometric transform family

`T_photo` must preserve geometry and point coordinates.

Candidate operations:

- brightness;
- contrast;
- saturation;
- random gamma;
- exposure shift;
- mild color temperature shift;
- Gaussian blur;
- mild motion blur;
- shot/Poisson-like noise;
- small Gaussian noise;
- salt/pepper noise;
- JPEG compression;
- mild haze/contrast washout.

Initial ranges:

- random scale: 0.75–2.0 before crop;
- brightness: ±0.20;
- contrast: ±0.20;
- saturation: ±0.15;
- blur probability: 0.20;
- gamma: 0.35–1.8, but extreme values low probability;
- exposure: roughly −2 to +1 stops, low probability for extremes;
- salt/pepper: about 0.001 initial.

Do not apply every degradation simultaneously.

---

## 14.2 Clean→degraded consistency

For a selected fraction of batches, generate a degraded view:

\[
x'=T_{photo}(x).
\]

Compute:

\[
D=f_\theta(x),
\qquad
D'=f_\theta(x').
\]

At each hierarchical scale:

\[
\mu^{(B)},\quad \mu'^{(B)}.
\]

Use clean prediction as detached teacher:

\[
\boxed{
L_{rob}=
\frac1{|\mathcal S|}
\sum_B
\operatorname{mean}_b
\operatorname{SmoothL1}
\left(
\log(1+\mu_b'^{(B)}),
\operatorname{sg}\left[\log(1+\mu_b^{(B)})\right]
\right).
}
\]

Recommended initial probability of running the second view:

\[
p_{rob}=0.30.
\]

The degraded view may also receive ordinary supervised losses with half weight. Compare:

- consistency only;
- supervised degraded view only;
- both.

No robustness branch is present at inference.

---

# 15. Total objective

Initial full criterion:

\[
\boxed{
L=
\lambda_HL_{HNB}
+\lambda_AL_{alloc}
+\lambda_NL_{HN}
+\lambda_EL_{empty}
+\lambda_GL_{global}
+\lambda_RL_{rob}.
}
\]

Starting coefficients:

| Term | Initial weight |
|---|---:|
| Hierarchical NB | 1.0 |
| Allocation | 0.5 |
| Hard negative | 0.25 |
| Empty image | 0.5 |
| Global count | 1.0 |
| Robust consistency | 0.1 |

These values are initial engineering defaults, **not claimed optima**.

Before extensive coefficient search, log the raw magnitude and gradient norm of every loss term. If one term is numerically dominant by orders of magnitude, fix normalization before tuning coefficients.

---

# 16. Curriculum schedule

Do not enable every auxiliary objective at epoch 1.

Let total training progress be \(t\in[0,1]\).

## Phase A — count stabilization, 0–10%

\[
L=L_{HNB}+L_{global}+0.25L_{alloc}.
\]

No hard-negative top-k and no clean/degraded consistency yet.

## Phase B — spatial + negative learning, 10–30%

\[
L=L_{HNB}+L_{global}+0.5L_{alloc}+0.1L_{HN}+0.25L_{empty}.
\]

## Phase C — full robust training, 30–100%

Use full objective and target weights.

Ablate curriculum versus all-loss-from-start. If no measurable benefit, remove curriculum from the final paper.

---

# 17. Dataset configuration

## 17.1 Initial training crops

| Dataset | Crop size | HNB scales | Allocation block |
|---|---:|---|---:|
| SHA | 448×448 | 16, 32, 64 | 16 |
| SHB | 448×448 | 16, 32, 64 | 16 |
| QNRF | 672×672 | 16, 32, 96 | 16 |
| NWPU | 672×672 | 16, 32, 96 | 16 |

Keep model architecture identical across datasets. Dataset-specific values should be limited mainly to:

- crop size;
- valid hierarchical scales;
- training-only block-count statistics;
- sampler statistics;
- possibly augmentation strength.

This supports the desired claim:

> **one architecture, dataset-adaptive training statistics.**

---

## 17.2 Point transformations

Every geometric augmentation must transform points exactly:

- resize/scaling;
- crop offset;
- horizontal flip.

Photometric transforms must not change point coordinates.

After crop, drop points outside valid crop bounds.

Use consistent `(x, y)` convention throughout the repository.

---

# 18. Density and luminance balancing

A small model can underfit rare hard regimes if trained with naive image-uniform sampling.

## 18.1 Training-only statistics

For each training image, cache:

- total count \(C_i\);
- image area;
- count per megapixel;
- mean or median luminance before augmentation.

Recommended image-level density scalar:

\[
d_i=\log\left(1+\frac{10^6C_i}{H_iW_i}\right).
\]

Create density quantile bins and luminance quantile bins using training images only.

## 18.2 Sampling weight

Let \(n_{g,l}\) be number of images in density/luminance group. Initial weight:

\[
\boxed{
w_i=\frac{1}{\sqrt{n_{g(i),l(i)}}}.
}
\]

Normalize weights before use in `WeightedRandomSampler`.

Do not use \(1/n\) initially because extremely rare groups may become excessively oversampled.

Ablate:

- uniform sampler;
- density-only sampler;
- density + luminance sampler.

---

# 19. NWPU-specific requirements

NWPU is a primary stress-test dataset, not an afterthought.

The final model must explicitly evaluate:

- negative / empty images;
- low-density scenes;
- medium density;
- very high density;
- extreme density;
- low-luminance subsets if the benchmark annotations/protocol permit it.

Do not report only overall MAE/RMSE.

At minimum produce internal diagnostic bins:

\[
C=0,
\]

\[
1\le C\le 10,
\]

\[
11\le C\le100,
\]

\[
101\le C\le1000,
\]

\[
C>1000.
\]

Also report:

- mean predicted count on GT-empty images;
- 95th percentile predicted count on GT-empty images;
- false-mass distribution across empty 16×16 blocks;
- MAE on top 10% densest validation images;
- error versus luminance quantile.

These diagnostics are important even if the official benchmark groups differ.

---

# 20. Variable-resolution inference policy

Main test inference is single-scale.

1. Normalize image with ImageNet statistics.
2. Pad image to the minimum multiple required by the backbone, preferably 16.
3. Run model once.
4. Crop the output map back to the valid stride-4 support corresponding to the original image.
5. Sum valid map values.

Do not resize every test image to the training crop unless the benchmark protocol explicitly requires it.

Avoid sliding-window inference in the main efficiency result. Use it only if an image cannot fit device memory, and report that protocol separately.

---

# 21. Training hyperparameters: initial implementation

Initial optimizer:

```text
AdamW
```

Suggested parameter groups:

- pretrained backbone LR: `2.5e-5`;
- new neck/head LR: `1e-4`;
- criterion dispersion parameters LR: `1e-4`;
- weight decay: `1e-4`.

If backbone underfits, ablate same LR for all parameters.

Schedule:

- 5% warm-up;
- cosine decay;
- approximately 300 epochs as first crowd-counting baseline, then standardize comparisons by optimizer steps.

Batch:

- target effective batch size = 8;
- use gradient accumulation if required.

Numerics:

- AMP allowed;
- NB NLL and `lgamma` path must use float32;
- gradient clipping at norm 1.0 initial;
- no silent NaN skipping.

Seeds:

- 3 seeds minimum for development ablations;
- 5 seeds preferred for final main result if compute permits.

Log:

- train loss terms separately;
- validation MAE/RMSE;
- gradient norm per loss occasionally;
- dispersion \(r_B\);
- hard-negative false mass;
- density-group errors;
- luminance-group errors.

---

# 22. Evaluation metrics

Standard:

\[
MAE=\frac1N\sum_i|\hat C_i-C_i|,
\]

\[
RMSE=\sqrt{\frac1N\sum_i(\hat C_i-C_i)^2}.
\]

Use NAE only according to benchmark convention.

Robustness diagnostics:

- empty-image MAE;
- empty-image predicted-count mean/p95;
- dense top-decile MAE;
- low-light quantile MAE;
- synthetic corruption relative degradation:

\[
\Delta MAE=MAE_{corrupt}-MAE_{clean}.
\]

Efficiency:

- total trainable parameters;
- deployed parameters;
- serialized size;
- MACs/FLOPs at declared resolution;
- peak memory;
- batch-1 latency;
- median and p90 latency;
- FPS;
- CPU and/or Jetson measurement if possible.

Do not call a model edge-ready from parameter count alone.

---

# 23. Parameter / compute target

The complete timm `mobilenetv4_conv_small_050` classification model is reported at about 2.2M parameters. HPC-Lite aims to use a trunk only through reduction 16 plus a very small neck/head.

Engineering target:

\[
\boxed{\text{deployed params}<1.5M}
\]

Stretch target:

\[
\boxed{\text{deployed params}\approx1.0-1.3M}
\]

These are **budgets, not current measured facts**.

The neck/head should ideally remain only tens of thousands of parameters. If the implementation exceeds the budget, inspect whether unused MobileNet stages remain registered before shrinking the 32-channel neck.

---

# 24. Accuracy targets: goals only

Do not place these values into an abstract or results table before experiments.

A useful strong target region for a ~1–1.5M model is approximately:

- SHA MAE: mid/high 50s or better;
- SHB MAE: about 6–7.5;
- QNRF MAE: low/mid 80s or better;
- NWPU: competitive overall plus clearly strong empty/extreme-density behavior.

The paper does **not** need absolute SOTA on every dataset if the model establishes a strong accuracy–efficiency–robustness Pareto point.

---

# 25. Lightweight comparison set

The final efficiency table should include, subject to reproducible code availability:

- SACC-Net-Light;
- LRMBNet variants;
- ZIP-P;
- ZIP-N;
- MobileCount;
- SANet;
- other recent <5M parameter crowd counters found in the pre-submission survey.

Also include larger accuracy-oriented references such as P2PNet, DM-Count-derived models, CLTR/PET/STEERER if useful, but do not call them lightweight.

**Critical fairness rule:** FLOPs from different resolutions are not directly comparable. Re-profile open-source models using the same profiler and declared input size whenever possible.

For each reported literature number, record:

- original source;
- input resolution;
- hardware if latency/FPS;
- whether value is reported by authors or re-measured by us.

---

# 26. Code-derived lessons that motivated the final method

## 26.1 SCALNet — false-positive suppression is implemented, not just theorized

Repo:

- https://github.com/WangyiNTU/SCALNet
- relevant file: `src/loc_loss.py`

Observed code behavior:

- modified focal loss downweights ordinary negatives;
- `_fp_loss` masks background and penalizes higher-confidence false positives;
- hierarchical heatmap supervision uses pooled targets at several scales.

What HPC-Lite keeps:

- explicit hard-negative emphasis;
- multi-scale supervision concept.

What HPC-Lite changes:

- uses false predicted count mass, not a fixed `pred > 0.1` threshold;
- no localization classifier at deployment.

## 26.2 Bayesian Crowd Counting — empty images get explicit zero-count treatment

Repo:

- https://github.com/zhiheng-ma/Bayesian-Crowd-Counting
- files: `losses/bay_loss.py`, `losses/post_prob.py`

Observed behavior:

- if image has no annotations, total predicted density is directly compared with a zero target;
- posterior target construction includes optional background component.

HPC-Lite keeps explicit whole-image empty suppression.

## 26.3 DM-Count — geometric loss is not sufficient for empty scenes

Repo:

- https://github.com/cvlab-stonybrook/DM-Count
- files: `losses/ot_loss.py`, `train_helper.py`

Observed behavior:

- OT computation runs only when the image has annotation points;
- global count loss remains important;
- geometry-aware training can be expensive with large point sets.

HPC-Lite therefore does not rely on OT as the core mechanism for NWPU negatives/extreme density.

## 26.4 S-DCNet — hierarchical local counting is practically useful

Repo:

- https://github.com/xhp-hust-2018-2011/S-DCNet
- files: `Network/SDCNet.py`, `Network/class_func.py`

Observed behavior:

- local counts are represented at multiple spatial resolutions;
- finer divisions are merged with learned weights;
- count classification limits open-set regression difficulty.

HPC-Lite keeps hierarchical local count supervision but eliminates multiple output branches and learned merging. All scales derive from one mass map by exact sum pooling.

## 26.5 P2PNet — useful point supervision but expensive structure for this objective

Repo:

- https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- files: `models/p2pnet.py`, `models/matcher.py`

Observed behavior:

- FPN uses 256-channel features;
- point regression/classification branches predict many anchors;
- Hungarian matching is used for one-to-one target assignment.

HPC-Lite keeps direct point-based target construction but avoids detection-style deployment complexity.

## 26.6 ZIP — useful lightweight decoder engineering

Repo:

- https://github.com/Yiming-M/ZIP
- files: `models/utils/refine.py`, `models/utils/upsample.py`, `configs/nwpu.yaml`

Observed behavior:

- lighter refinement uses depthwise 3×3 + pointwise 1×1 + residual;
- NWPU training includes scale, brightness, contrast, blur, and sparse noise augmentation.

These are incorporated as engineering guidance.

---

# 27. Relationship to PML, ZIP, OT, and other previous ideas

PML and ZIP are no longer mandatory components.

## PML

Repo:

- https://github.com/Elin24/pml

Use as optional geometry-supervision ablation:

```text
HPC-Lite + PML
```

Reason it is not core:

- point-neighbor geometry can become costly as point count grows;
- NWPU includes extremely large annotation sets;
- core method must already solve negatives and extreme counts without relying on PML.

## ZIP

Repo:

- https://github.com/Yiming-M/ZIP

Use as:

- direct baseline;
- optional training-only auxiliary count head if later justified.

Do not add it by default simply because earlier proposal was PML–ZIP.

## DM-Count OT

Optional training-only geometry ablation:

```text
HPC-Lite + OT
```

Keep out of core until training-time and accuracy benefit are measured.

## Characteristic-function loss

Reference code:

- https://github.com/wbshu/Crowd_Counting_in_the_Frequency_Domain

Possible future annotation-noise ablation. Not core.

---

# 28. Mandatory ablation ladder

Keep ablations incremental and attributable.

| ID | Variant | Purpose |
|---|---|---|
| A0 | backbone + tiny neck + Softplus map + global count | pure architecture baseline |
| A1 | A0 + hierarchical Poisson | test hierarchy alone |
| A2 | A0 + hierarchical NB | test overdispersion model |
| A3 | A2 + density-stratified NB reduction | test imbalance handling |
| A4 | A3 + local allocation | test point-derived spatial supervision |
| A5 | A4 + hard-negative mining | test background false-positive reduction |
| A6 | A5 + empty-image loss | test NWPU negative scenes |
| A7 | A6 + density/luminance sampler | test regime balancing |
| A8 | A7 + adverse-condition training | full HPC-Lite |

Main proposed method initially:

\[
\boxed{A8}
\]

External auxiliary ablations:

| ID | Variant |
|---|---|
| X1 | A8 + PML |
| X2 | A8 + ZIP auxiliary |
| X3 | A8 + DM-Count OT |
| X4 | A8 + teacher distillation |

If an auxiliary does not consistently help across seeds/datasets, remove it from the final method.

---

# 29. Architecture ablations

Backbones:

1. MobileNetV4-Conv-Small-0.5 — main;
2. MobileNetV4-Conv-Small — accuracy-oriented;
3. RepViT-M0.6/M0.9 — latency-oriented comparison if implementation is reliable.

Neck width:

\[
24,32,48.
\]

Context:

- no context;
- dilation {1,2};
- dilation {1,2,3}.

Fusion:

- additive;
- concat only as a cost/accuracy ablation.

Output activation:

- Softplus main;
- ReLU ablation.

---

# 30. Statistical ablations

Probabilistic family:

1. Poisson;
2. Negative Binomial;
3. only if diagnostics justify it: Hurdle-NB or ZINB.

Do **not** introduce ZINB just because zeros exist. First compare empirical zero frequency with that predicted by fitted NB. If NB already models zeros adequately, extra zero-inflation parameters are unnecessary.

Possible future model-selection diagnostic for each scale:

- empirical mean;
- empirical variance;
- empirical zero rate;
- Poisson expected zero rate;
- NB fitted zero rate.

If zero excess remains severe after NB, then test a hurdle/ZINB criterion as a separate experiment.

---

# 31. Hard-condition ablations

NWPU-focused:

- HN top fraction 5/10/20%;
- empty loss off/on;
- density sampler off/on;
- luminance sampler off/on;
- robust second-view probability 0/0.3/0.5;
- gamma augmentation off/on;
- blur/noise augmentation off/on.

Report overall metrics and the affected subgroup metrics. A component should not be retained solely because it improves one subgroup while seriously hurting overall performance unless the paper explicitly targets that tradeoff.

---

# 32. Failure criteria

The design should be reconsidered if any of the following occurs after reasonable tuning:

1. hierarchical NB does not beat hierarchical Poisson or ordinary count supervision;
2. hard-negative mining improves empty scenes but significantly increases undercount on sparse positives;
3. local allocation harms dense-count performance consistently;
4. robustness training reduces clean-set accuracy more than its corruption benefit;
5. final deployed params remain >2M without a major accuracy advantage over lightweight baselines;
6. NWPU densest-group errors remain catastrophically high;
7. multi-dataset results require substantially different architectures rather than training statistics.

A negative result is preferable to keeping unsupported complexity.

---

# 33. Repository layout

Recommended structure:

```text
hpc_lite/
├── README.md
├── requirements.txt
├── configs/
│   ├── sha.yaml
│   ├── shb.yaml
│   ├── qnrf.yaml
│   └── nwpu.yaml
├── hpc/
│   ├── models/
│   │   ├── hpc_lite.py
│   │   ├── backbone.py
│   │   ├── neck.py
│   │   └── blocks.py
│   ├── losses/
│   │   ├── negative_binomial.py
│   │   ├── allocation.py
│   │   ├── hard_negative.py
│   │   ├── robustness.py
│   │   └── criterion.py
│   ├── targets/
│   │   ├── block_counts.py
│   │   └── allocation_target.py
│   ├── data/
│   │   ├── common.py
│   │   ├── sha.py
│   │   ├── qnrf.py
│   │   ├── nwpu.py
│   │   ├── transforms.py
│   │   └── sampler.py
│   ├── metrics/
│   │   ├── counting.py
│   │   └── subgroup.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── checkpoint.py
├── tools/
│   ├── compute_dataset_stats.py
│   ├── profile_model.py
│   ├── export_onnx.py
│   └── analyze_errors.py
├── train.py
├── evaluate.py
└── tests/
    ├── test_model_shapes.py
    ├── test_block_counts.py
    ├── test_allocation_mass.py
    ├── test_nb_loss.py
    ├── test_hard_negative.py
    ├── test_empty_batch.py
    └── test_count_conservation.py
```

---

# 34. Configuration schema

Example NWPU configuration:

```yaml
experiment:
  name: hpc_lite_nwpu
  seed: 42

dataset:
  name: nwpu
  root: /path/to/nwpu
  crop_size: 672
  output_stride: 4
  hnb_blocks: [16, 32, 96]
  allocation_block: 16

model:
  backbone: mobilenetv4_conv_small_050
  pretrained: true
  feature_reductions: [4, 8, 16]
  neck_width: 32
  context_dilations: [1, 2, 3]
  head_eps: 1.0e-6
  data_driven_head_bias: true

loss:
  hnb: 1.0
  allocation: 0.5
  hard_negative: 0.25
  empty: 0.5
  global_count: 1.0
  robust: 0.1
  hard_negative_fraction: 0.10
  density_stratified_nb: true

robustness:
  second_view_prob: 0.30
  brightness: 0.20
  contrast: 0.20
  saturation: 0.15
  blur_prob: 0.20
  gamma_min: 0.35
  gamma_max: 1.80
  salt_pepper: 0.001

sampler:
  weighted: true
  density_bins: 5
  luminance_bins: 4
  exponent: 0.5

optimizer:
  name: adamw
  backbone_lr: 2.5e-5
  head_lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0

schedule:
  epochs: 300
  warmup_fraction: 0.05
  cosine: true

training:
  effective_batch_size: 8
  amp: true
  num_workers: 8
  save_best_by: mae
```

Agents must not assume these values are final; all tuned values must be traceable in experiment configs.

---

# 35. Mandatory unit tests

## T1 — backbone reductions

For arbitrary valid input:

```text
C4 ≈ H/4
C8 ≈ H/8
C16 ≈ H/16
D ≈ H/4
```

Exact behavior under padding must be documented.

## T2 — global count target

For any point list:

\[
C=N.
\]

## T3 — exact block-count conservation

For each configured block size on divisible crops:

\[
\sum_bY_b^{(B)}=N.
\]

## T4 — allocation target conservation

For each allocation block:

\[
\sum_kZ_{bk}=Y_b^{(16)}
\]

within tolerance `1e-5`.

Global:

\[
\sum_{b,k}Z_{bk}=N.
\]

## T5 — border points

Test points at/near:

- `(0,0)`;
- right/bottom image edge;
- 16-pixel block boundary;
- crop boundary.

Mass must not disappear or leak to a neighboring allocation block.

## T6 — sum pooling

For every `D`:

\[
\sum_b\mu_b^{(B)}=\sum_{ij}D_{ij}
\]

on divisible crops.

## T7 — NB stability

Finite loss and finite gradients for target counts through 20,000.

## T8 — all-empty batch

No division by zero, no NaN, allocation loss exactly zero.

## T9 — no-zero-block batch

Hard-negative loss safely returns zero.

## T10 — degradation geometry

Photometric-only robustness transformation must preserve target point coordinates exactly.

## T11 — backward test

One complete criterion backward pass must succeed in AMP training mode.

## T12 — parameter budget test

Automated profiler writes:

- total params;
- trainable params;
- deploy params;
- per-module params.

Do not hard-fail CI at <1.5M until the backbone is physically truncated, but flag regressions.

---

# 36. Integration sanity experiment before full training

Before any 300-epoch experiment, overfit a tiny dataset:

- 8–16 crops;
- include at least one empty crop;
- one sparse crop;
- one dense crop.

Expected behavior:

- total training loss decreases;
- predicted empty count approaches zero;
- dense crop predicted count approaches GT;
- NB dispersion remains finite;
- allocation map concentrates around point locations;
- no term produces NaN.

If the model cannot overfit this tiny set, full-dataset training is prohibited until fixed.

---

# 37. Experiment logging contract

Every run must save:

```text
config.yaml
commit_sha.txt
environment.txt
train.csv
val.csv
best.pt
last.pt
profile.json
```

`profile.json` should include:

```json
{
  "params_total": null,
  "params_deploy": null,
  "macs_resolution": null,
  "macs": null,
  "latency_device": null,
  "latency_batch": 1,
  "latency_median_ms": null,
  "latency_p90_ms": null,
  "peak_memory_mb": null
}
```

Do not manually copy result values between experiments.

---

# 38. Agent work decomposition

## Agent A — model architecture

Own:

- `models/backbone.py`
- `models/blocks.py`
- `models/neck.py`
- `models/hpc_lite.py`

Deliverables:

1. verified reduction-4/8/16 feature extraction;
2. additive 32-channel neck;
3. Softplus mass head;
4. data-driven head-bias initializer;
5. parameter report;
6. shape unit tests.

Acceptance:

- forward works for 448 and 672 crops;
- arbitrary validation-size padded forward works;
- count = exact sum of valid `D`;
- no unused classifier;
- report whether reduction-32 layers still exist in model parameters.

## Agent B — targets and probabilistic losses

Own:

- exact integer block counts;
- block-constrained allocation target;
- NB NLL;
- dispersion initialization;
- density-stratified block reduction.

Acceptance:

- all conservation tests pass;
- NB finite through y=20,000;
- no GT leakage across allocation blocks.

## Agent C — hard negatives and robustness

Own:

- per-image top-k zero-block mining;
- whole-image empty loss;
- photometric transforms;
- clean→degraded consistency.

Acceptance:

- no additional inference modules;
- no geometry change from photometric transforms;
- zero/no-zero edge cases handled.

## Agent D — datasets and sampling

Own:

- SHA/SHB/QNRF/NWPU point loaders;
- exact resize/crop/flip transforms;
- training-only statistics;
- density/luminance weighted sampler.

Acceptance:

- point count before/after transforms validated;
- no validation/test data used for statistics;
- cached statistics are deterministic and versioned.

## Agent E — trainer/evaluator/profiler

Own:

- optimizer groups;
- curriculum;
- AMP float32 NB path;
- checkpointing;
- subgroup metrics;
- same-resolution efficiency profiler;
- ONNX/export checks.

Acceptance:

- full tiny-set overfit works;
- 1-epoch smoke training on each dataset works;
- profiler generates reproducible JSON.

## Agent F — baseline reproduction and literature

Own:

- lightweight baseline list;
- original paper/code verification;
- same-resolution re-profiling where possible;
- pre-submission novelty search.

Must maintain a table with columns:

```text
method | paper | repo | params_reported | params_reprofiled |
input_resolution | FLOPs_reported | FLOPs_reprofiled |
SHA | SHB | QNRF | NWPU | hardware | notes
```

Never mix incomparable FLOP resolutions without an explicit note.

---

# 39. Recommended implementation order

Do not parallelize everything before interfaces stabilize.

## Milestone 1 — architecture baseline

- backbone + tiny neck + output map;
- global count loss only;
- profiler.

## Milestone 2 — exact target infrastructure

- block count targets;
- allocation targets;
- conservation tests.

## Milestone 3 — hierarchical probability

- Poisson baseline;
- NB implementation;
- dispersion statistics;
- density-stratified reduction.

## Milestone 4 — negative robustness

- hard-zero mining;
- empty loss;
- NWPU subgroup diagnostics.

## Milestone 5 — adverse conditions

- degradation pipeline;
- consistency training;
- luminance diagnostics.

## Milestone 6 — full ablation

Only after A0–A8 are stable should PML/ZIP/OT/KD auxiliary experiments begin.

---

# 40. First experiments to run

Recommended order to reduce wasted compute:

1. SHB smoke test — sparse/background behavior is easy to inspect.
2. SHA — dense but manageable.
3. NWPU — primary negative/extreme-density stress test.
4. QNRF — dense/high-resolution confirmation.

For each dataset:

1. overfit tiny subset;
2. 10-epoch diagnostic run;
3. A0 vs A2 (global vs hierarchical NB);
4. A2 vs A5/A6 on false positives;
5. only then run full A8.

---

# 41. Novelty positioning

Do not claim that the following concepts are individually new:

- hierarchical counting;
- density-aware learning;
- hard-negative mining;
- Negative-Binomial count modeling;
- photometric augmentation;
- multi-scale supervision;
- lightweight depthwise convolution.

Potential contribution is their **crowd-counting formulation and lightweight integration around one conserved mass map**, if experiments validate it.

A cautious working contribution statement:

> We propose a lightweight single-map crowd-counting framework in which one nonnegative mass field is trained through count-conserving hierarchical probabilistic supervision, locally normalized point-derived spatial allocation, explicit hard-background mass mining, and adverse-condition consistency, while preserving a minimal single-head inference graph.

Before submission, run a dedicated novelty search for combinations of:

- hierarchical Negative-Binomial crowd counting;
- single-map count-conserving multi-scale likelihood;
- hard-negative mass mining for crowd counting;
- point-derived normalized allocation without Gaussian targets;
- lightweight robust crowd counting.

Use “to the best of our knowledge” only after that search.

---

# 42. Paper narrative if experiments succeed

The paper should tell a simple story:

1. lightweight models have limited capacity;
2. crowd scenes span empty backgrounds to >10k people and adverse illumination;
3. one supervision rule is inefficient across all regimes;
4. instead of increasing architecture complexity, teach one mass map with structured training signals;
5. hierarchical NB handles count scale/overdispersion;
6. normalized local allocation gives point-level spatial guidance without Gaussian density maps;
7. hard-zero mining and explicit empty loss handle hallucinations;
8. robustness training handles dark/noisy conditions;
9. all extra machinery disappears at deployment;
10. evaluate accuracy, efficiency, empty scenes, extreme density, and low illumination.

---

# 43. Result tables to prepare

## Table A — main accuracy

```text
Method | Params | FLOPs | SHA MAE/RMSE | SHB | QNRF | NWPU
```

## Table B — robustness / NWPU

```text
Method | Empty MAE | Empty Pred Mean | Empty Pred P95 |
C>1000 MAE | Top-10%-density MAE | Dark-bin MAE
```

## Table C — efficiency

```text
Method | Params | Model MB | Input | MACs | Median ms | P90 ms | FPS | Peak MB
```

## Table D — ablation

```text
HNB | Stratified | Alloc | HN | Empty | Robust | MAE | RMSE | Empty MAE | Dense MAE
```

## Table E — probabilistic choice

```text
Poisson | NB | Hurdle-NB(if tested) | MAE | Dense MAE | NLL | calibration diagnostics
```

---

# 44. Deployment protocol

After final FP32 accuracy is stable:

1. export ONNX;
2. check numerical count parity on 100 images;
3. benchmark FP32 and FP16;
4. optionally test INT8 post-training quantization;
5. do not include quantization in the main algorithmic claim unless fully evaluated.

Latency methodology:

- batch 1;
- fixed declared resolution for comparisons;
- 50+ warmup iterations;
- 200+ measured iterations;
- synchronize GPU before/after timing;
- report median and p90;
- report device and software stack.

---

# 45. Reproducibility rules

- all dataset statistics are training-only;
- all validation splits are fixed and versioned;
- no hyperparameter tuning on NWPU test server;
- seeds stored in config;
- exact package versions stored;
- every paper result links to an experiment ID/checkpoint;
- reported parameter/FLOP numbers come from scripts, not hand calculations;
- mark literature-reported versus re-profiled efficiency values separately.

For datasets without a public validation set, use a fixed train split such as 90/10 stratified by log-count quantiles and never change it during ablation.

---

# 46. Immediate technical questions agents must resolve experimentally

1. Does `mobilenetv4_conv_small_050` physically truncate below 1.5M params at reduction 16?
2. Is 32-channel neck enough, or does 48 materially improve QNRF/NWPU?
3. Is NB consistently better than Poisson after stratified reduction?
4. Does allocation block 16 remain stable in extremely dense QNRF/NWPU crops?
5. Does hard-negative top-k reduce false mass without sparse undercount?
6. Does explicit empty-image loss add benefit beyond NB zeros + HN?
7. Does density/luminance sampling help or overfit rare cases?
8. Is robustness consistency useful beyond strong supervised augmentation?
9. Are dilation branches useful relative to their small compute cost?
10. Does the final model establish a useful Pareto point against SACC-Light/ZIP-P/ZIP-N/LRMBNet-type baselines?

No theoretical preference overrides these measurements.

---

# 47. Optional future branches — only after core is complete

Priority order:

### F1 — Hurdle-NB / ZINB

Only if NB zero-rate calibration is inadequate.

### F2 — Dirichlet-Multinomial spatial allocation

Only if simple normalized allocation fails specifically on heavily clustered dense blocks.

### F3 — teacher distillation

Train-only large teacher; deployment student unchanged.

### F4 — PML auxiliary

Test whether geometry supervision improves localization/counting enough to justify training cost.

### F5 — Unbalanced OT

Train-only geometric mass alignment; consider only with efficient implementation.

### F6 — characteristic-function annotation-noise objective

Useful if annotation uncertainty/noise becomes a central paper claim.

### F7 — quantization-aware training

Deployment optimization, not core research novelty.

---

# 48. Things agents must not do

- Do not silently restore the old PML–ZIP architecture.
- Do not add a query decoder because P2PNet/DETR is familiar.
- Do not generate Gaussian density maps for the core target pipeline.
- Do not use bilinear-splatted maps as integer NB count targets.
- Do not drop the last partial block silently during pooling.
- Do not report a guessed 1.2M parameter count.
- Do not compare our 512×512 FLOPs with another paper's 224×224 FLOPs as if equivalent.
- Do not claim robustness based only on augmentation without subgroup/corruption evaluation.
- Do not claim edge deployment using only Params/FLOPs.
- Do not tune on test sets.
- Do not keep optional modules merely because they make the method look more sophisticated.

---

# 49. Minimum definition of done

The core research implementation is considered complete when all of the following are true:

1. all unit tests pass;
2. tiny-set overfit passes;
3. one model architecture runs unchanged on SHA/SHB/QNRF/NWPU;
4. exact block count conservation is verified;
5. NB is numerically stable up to 20k targets;
6. empty-scene diagnostics are implemented;
7. dense-scene diagnostics are implemented;
8. A0–A8 ablation can be run from configs without code edits;
9. deploy parameter count and same-resolution MACs are measured;
10. single-scale inference can be exported;
11. at least three seeds exist for the main comparison;
12. baseline efficiency values are source-verified or re-profiled;
13. pre-submission novelty search is completed.

---

# 50. Reference links

## Core implementation / lightweight

- MobileNetV4 paper: https://arxiv.org/abs/2404.10518
- timm: https://github.com/huggingface/pytorch-image-models
- MobileNetV4-Small-0.5 timm model card: https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
- ZIP: https://github.com/Yiming-M/ZIP

## Crowd supervision / negative / hierarchy

- SCALNet: https://github.com/WangyiNTU/SCALNet
- Bayesian Crowd Counting: https://github.com/zhiheng-ma/Bayesian-Crowd-Counting
- DM-Count: https://github.com/cvlab-stonybrook/DM-Count
- S-DCNet: https://github.com/xhp-hust-2018-2011/S-DCNet
- P2PNet: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- PML: https://github.com/Elin24/pml
- Crowd Counting in the Frequency Domain: https://github.com/wbshu/Crowd_Counting_in_the_Frequency_Domain

## Optional optimization / broader ML background

- PCGrad: https://arxiv.org/abs/2001.06782
- MobileNetV4: https://arxiv.org/abs/2404.10518

---

# 51. Final frozen core specification

Unless an experiment disproves it, agents should implement the following first:

\[
\boxed{
\begin{aligned}
&\textbf{Backbone: }MobileNetV4\text{-Conv-Small-0.5, truncated to /16 if possible}\\
&\textbf{Features: }1/4,1/8,1/16\\
&\textbf{Neck: }32\text{-channel additive depthwise-separable FPN}\\
&\textbf{Context: }DWConv\ d=1,2,3\text{ at /16}\\
&\textbf{Output: }1\text{-channel Softplus count-mass map at stride 4}\\
&\textbf{Inference count: }\hat C=\sum D\\
&\textbf{Counting supervision: hierarchical density-stratified Negative Binomial}\\
&\textbf{Spatial supervision: block-constrained normalized point allocation}\\
&\textbf{Negative supervision: per-image top-k zero-block false-mass mining}\\
&\textbf{Empty supervision: explicit whole-image zero-mass loss}\\
&\textbf{Robustness: clean→degraded local-count consistency, training only}\\
&\textbf{Sampling: training-only density+luminance balancing}\\
&\textbf{Inference auxiliary heads: none}.
\end{aligned}
}
\]

This is the implementation baseline. PML, ZIP, OT, KD, hurdle models, and more complex modules are **research extensions**, not prerequisites.

---

## End-of-document checklist for an implementing agent

Before writing code, answer yes to all:

- [ ] I understand `D` is count mass per stride-4 cell.
- [ ] I will build exact integer block-count targets directly from point coordinates.
- [ ] I will build a separate block-constrained soft allocation target.
- [ ] I will not let allocation splatting leak mass across a 16×16 allocation block.
- [ ] I will use block scales that divide the training crop.
- [ ] I will keep NB math in float32.
- [ ] I will initialize the Softplus head to a realistic small mass.
- [ ] I will mine hard zero blocks per image, not globally only.
- [ ] I will test all-empty and all-positive batches.
- [ ] I will not use test data to compute quantiles/statistics.
- [ ] I will profile the real deploy graph before claiming parameter/FLOP values.
- [ ] I will keep the inference graph single-head and single-scale.

