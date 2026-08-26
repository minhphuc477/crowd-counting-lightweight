# HPC-Lite — Objective & Training Revision Plan (Batch 16, 2000 Epochs)

## 0. Mục tiêu của tài liệu

Tài liệu này là **implementation guide** để agent tiếp theo sửa code HPC-Lite theo hướng ưu tiên **giảm MAE thực tế** trước khi thay kiến trúc lần nữa.

Mục tiêu chính:

1. Giữ nguyên architecture của model đang train để cô lập ảnh hưởng của loss/training.
2. Thêm một **direct count objective** có gradient không biến mất khi crowd count lớn.
3. Hạ vai trò của HNB từ objective chính xuống probabilistic regularizer.
4. Giảm dominance của allocation loss vì allocation không kiểm soát tổng mass.
5. Freeze dispersion trước để NB không “explain away” count error bằng uncertainty.
6. Chỉnh optimizer để pretrained lightweight backbone thật sự thích nghi với crowd counting.
7. Sau khi xác nhận objective mới tốt hơn, mới sửa data pipeline và architecture.
8. Batch size cố định: **16**.
9. Training budget cố định: **2000 epochs**.

---

# 1. Chẩn đoán hiện tại

## 1.1. Output của HPC-Lite

Model sinh một positive count-mass map stride 4:

\[
D \in \mathbb{R}_{+}^{H/4 \times W/4}
\]

và count ảnh:

\[
\hat C = \sum_{u,v} D_{u,v}.
\]

Head hiện tại:

```python
h = self.head_act(self.head_norm(self.head_dw(p4)))
z = self.head_out(h)
d = F.softplus(z) + self.eps_d
```

Không thay đổi head trong experiment đầu tiên.

---

# 2. Vì sao objective hiện tại có thể cho loss đẹp nhưng MAE cao

## 2.1. HNB gradient yếu dần ở dense counts

Negative Binomial NLL:

\[
\mathcal L_{NB}(y,\mu,r)
= -\log P(Y=y\mid \mu,r).
\]

Với parameterization hiện tại, derivative theo predicted mean là:

\[
\frac{\partial \mathcal L_{NB}}{\partial \mu}
=
\frac{r(\mu-y)}{\mu(r+\mu)}.
\]

Khi \(y\) và \(\mu\) lớn:

\[
\left|\frac{\partial \mathcal L_{NB}}{\partial \mu}\right|
\approx O\left(\frac{1}{y}\right)
\]

hoặc nhỏ hơn tùy residual.

Do đó NB rất hợp để model probabilistic count distribution nhưng **không tối ưu trực tiếp MAE**.

---

## 2.2. Learnable dispersion có thể hấp thụ error

Hiện tại:

\[
r_B = \operatorname{softplus}(\eta_B)+10^{-4}.
\]

Nếu \(r_B\) được learn tự do, optimizer có hai cách giảm NLL:

1. sửa \(\mu\) gần \(y\);
2. thay đổi dispersion để distribution rộng hơn.

Benchmark MAE chỉ quan tâm:

\[
|\hat C-C|.
\]

Vì vậy experiment đầu tiên nên **freeze dispersion** để buộc model sửa mean prediction.

---

## 2.3. Allocation loss không kiểm soát block count magnitude

Current allocation:

\[
p_{bk}
=
\frac{D_{bk}+\epsilon}
{\mu_b+K\epsilon},
\quad
\mu_b=\sum_k D_{bk}.
\]

Loss:

\[
\mathcal L_{alloc,b}
=
-\frac1{y_b}
\sum_k Z_{bk}\log(p_{bk}+\epsilon).
\]

Nếu toàn block bị scale:

\[
D'_{bk}=aD_{bk},\quad a>0,
\]

thì gần như:

\[
p'_{bk}=p_{bk}.
\]

Do đó allocation có thể đúng spatial distribution nhưng block mass vẫn sai.

**Kết luận:** allocation phải là spatial regularizer, không phải main objective.

---

## 2.4. Log-count global loss làm gradient dense scenes quá nhỏ

Current default:

\[
\mathcal L_{global}^{log}
=
SmoothL1(\log(1+\hat C),\log(1+C)).
\]

Gradient chứa factor:

\[
\frac{1}{1+\hat C}.
\]

Khi count lớn, global signal yếu dần.

Trong khi benchmark MAE đánh trực tiếp:

\[
\frac1N\sum_i|\hat C_i-C_i|.
\]

Ta cần một objective aligned trực tiếp với metric.

---

# 3. Objective mới — recommended baseline

## 3.1. Direct normalized MAE loss

Định nghĩa:

\[
\boxed{
\mathcal L_{count}
=
\frac{1}{S_C}
\frac1B
\sum_i |\hat C_i-C_i|
}
\]

với:

\[
\boxed{S_C=100}.
\]

Tại sao chia 100?

- giữ numerical scale của loss quanh 0–2 thay vì 0–200;
- gradient theo mỗi cell không phụ thuộc crowd count;
- nếu MAE = 80 thì \(L_{count}\approx0.8\), rất dễ cân bằng với HNB.

Gradient theo một mass cell:

\[
\frac{\partial L_{count}}{\partial D_{uv}}
=
\frac{\operatorname{sign}(\hat C-C)}{100B}.
\]

Không có factor \(1/(1+C)\).

### Code suggestion

File đề nghị: `losses/hard_negative.py` hoặc tách mới `losses/counting_losses.py` nếu repo đã clean module structure.

```python
class DirectCountL1Loss(nn.Module):
    """Direct image-count objective aligned with MAE."""

    def __init__(self, count_scale: float = 100.0):
        super().__init__()
        if count_scale <= 0:
            raise ValueError("count_scale must be > 0")
        self.count_scale = float(count_scale)

    def forward(self, d_map: torch.Tensor, gt_counts: torch.Tensor) -> torch.Tensor:
        pred_counts = d_map.sum(dim=(-1, -2, -3))
        gt_counts = gt_counts.to(pred_counts.device, dtype=pred_counts.dtype)
        return torch.mean(torch.abs(pred_counts - gt_counts)) / self.count_scale
```

### Diagnostic phải log thêm

```python
with torch.no_grad():
    pred_counts = d_map.sum(dim=(-1, -2, -3))
    raw_batch_mae = torch.mean(torch.abs(pred_counts - gt_counts))
```

Log:

```text
loss_count
batch_count_mae
mean_pred_count
mean_gt_count
mean_signed_count_error
```

---

# 4. Total loss mới

## 4.1. Recommended Phase-1 objective

Experiment đầu tiên dùng:

\[
\boxed{
L =
1.0L_{count}
+0.35L_{HNB}
+0.15L_{alloc}
+0.10L_{HN}
+0.25L_{empty}
+0.10L_{global-log}
+0.05L_{rob}
}
\]

Recommended coefficients:

```yaml
lambda_count: 1.00
lambda_hnb: 0.35
lambda_alloc: 0.15
lambda_hn: 0.10
lambda_empty: 0.25
lambda_global_log: 0.10
lambda_rob: 0.05
count_scale: 100.0
```

### Tại sao không bỏ HNB?

HNB vẫn giữ:

- hierarchical local count calibration;
- zero/low/high count structure;
- probabilistic robustness;
- contribution riêng của HPC-Lite.

Nhưng HNB không còn là force lớn nhất.

### Tại sao không bỏ global log hoàn toàn?

Giữ weight nhỏ 0.10 để bảo tồn stability khi count residual rất lớn lúc đầu training. Nhưng direct count loss mới là objective chính.

---

# 5. Criterion patch

File: `losses/criterion.py`

## 5.1. Import DirectCountL1Loss

```python
from .hard_negative import (
    HardNegativeMassLoss,
    WholeImageEmptyLoss,
    GlobalCountLoss,
    DirectCountL1Loss,
)
```

## 5.2. Constructor mới

Thêm:

```python
lambda_count: float = 1.0,
count_scale: float = 100.0,
```

Store:

```python
self.lambda_count = float(lambda_count)
```

Instantiate:

```python
self.count_loss = DirectCountL1Loss(count_scale=count_scale)
```

## 5.3. Forward

Sau global loss:

```python
l_count = self.count_loss(d_map, gt_counts)
loss_dict["loss_count"] = l_count.detach()

with torch.no_grad():
    pred_counts = d_map.sum(dim=(-1, -2, -3))
    gt_counts_dev = gt_counts.to(pred_counts.device, dtype=pred_counts.dtype)
    loss_dict["batch_count_mae"] = torch.mean(
        torch.abs(pred_counts - gt_counts_dev)
    )
    loss_dict["mean_pred_count"] = pred_counts.mean()
    loss_dict["mean_gt_count"] = gt_counts_dev.mean()
    loss_dict["mean_signed_count_error"] = (
        pred_counts - gt_counts_dev
    ).mean()
```

## 5.4. Bỏ curriculum hard-code cũ

Không dùng phase mà weight phase A/B hard-code độc lập với lambda.

Dùng factor schedule nhân với base lambda.

Recommended:

```python
if not self.enable_curriculum:
    f_count = f_hnb = f_alloc = f_hn = f_empty = f_global = f_rob = 1.0
else:
    if progress < 0.05:
        # 0-5%: count stabilization
        f_count = 1.0
        f_hnb = 0.5
        f_alloc = 0.0
        f_hn = 0.0
        f_empty = 0.5
        f_global = 1.0
        f_rob = 0.0
    elif progress < 0.15:
        # 5-15%: introduce local spatial learning
        f_count = 1.0
        f_hnb = 1.0
        f_alloc = 0.5
        f_hn = 0.5
        f_empty = 1.0
        f_global = 1.0
        f_rob = 0.0
    else:
        # 15-100%: full objective
        f_count = 1.0
        f_hnb = 1.0
        f_alloc = 1.0
        f_hn = 1.0
        f_empty = 1.0
        f_global = 1.0
        f_rob = 1.0
```

Then:

```python
w_count = self.lambda_count * f_count
w_hnb = self.lambda_hnb * f_hnb
w_alloc = self.lambda_alloc * f_alloc
w_hn = self.lambda_hn * f_hn
w_empty = self.lambda_empty * f_empty
w_global = self.lambda_global * f_global
w_rob = self.lambda_rob * f_rob
```

Total:

```python
total_loss = (
    w_count * l_count
    + w_hnb * l_hnb
    + w_alloc * l_alloc
    + w_hn * l_hn
    + w_empty * l_empty
    + w_global * l_global
    + (w_rob * l_rob if d_degraded is not None else 0.0)
)
```

**Acceptance test:** set every lambda to zero => `loss_total == 0` at every progress value.

---

# 6. Freeze NB dispersion — experiment bắt buộc

File: `losses/negative_binomial.py`

## 6.1. Recommended behavior

Trong run Phase-1:

\[
\boxed{r_B\text{ fixed for whole training}}
\]

Không train `raw_dispersions`.

Có hai implementation options.

### Option A — simplest

Sau data-stat initialization:

```python
for p in criterion.hnb_loss.raw_dispersions.parameters():
    p.requires_grad_(False)
```

Optimizer không được chứa các parameter này.

### Option B — config clean hơn

Thêm argument:

```python
learn_dispersion: bool = False
```

Trong `HierarchicalNBLoss.__init__`:

```python
self.learn_dispersion = bool(learn_dispersion)

if not use_poisson:
    self.raw_dispersions = nn.ParameterDict({
        str(b): nn.Parameter(
            torch.tensor(inv_softplus(10.0), dtype=torch.float32),
            requires_grad=self.learn_dispersion,
        )
        for b in self.block_sizes
    })
```

## 6.2. Stable inverse softplus bắt buộc

```python
def inv_softplus(y: float) -> float:
    y = max(float(y), 1e-8)
    if y > 20.0:
        return y
    return math.log(math.expm1(y))
```

## 6.3. Clamp MoM initialization

Để tránh dispersion cực lớn khi `var ≈ mean`:

```python
r0 = max((mean * mean) / max(var - mean, 1e-6), 1e-3)
r0 = min(r0, 100.0)
```

Recommended bounds:

\[
\boxed{1 \le r_B \le 100}
\]

Không cần ép lower bound bằng code nếu statistics hợp lý; nhưng log giá trị thật.

---

# 7. Allocation loss — giữ nhưng giảm dominance

Không thay công thức trong experiment đầu tiên.

Weight:

\[
0.5 \rightarrow \boxed{0.15}.
\]

Phải log thêm intrinsic entropy và KL excess để tránh hiểu sai raw CE.

## 7.1. Diagnostic

Với:

\[
q_{bk}=\frac{Z_{bk}}{y_b}
\]

thì:

\[
L_{alloc}=H(q)+KL(q\|p).
\]

Code diagnostic:

```python
q_pos = z_pos / y_pos.unsqueeze(-1).clamp_min(1.0)
q_pos = q_pos.clamp_min(0.0)

entropy = -(q_pos * torch.log(q_pos.clamp_min(self.eps))).sum(dim=-1)
ce = -(q_pos * torch.log(p_pos.clamp_min(self.eps))).sum(dim=-1)
kl = (ce - entropy).clamp_min(0.0)
```

Log:

```text
alloc_ce
alloc_entropy
alloc_kl
```

**Không dùng `loss_alloc -> 0` làm convergence criterion.**

---

# 8. Optimizer — Batch 16, backbone phải được fine-tune đủ mạnh

## 8.1. Recommended Phase-1 optimizer

Dùng AdamW:

```yaml
optimizer: AdamW
batch_size: 16
epochs: 2000
base_lr: 1.0e-4
weight_decay: 1.0e-4
grad_clip: 1.0
```

Trong controlled run đầu tiên:

\[
\boxed{LR_{backbone}=LR_{neck}=LR_{head}=10^{-4}}
\]

Không dùng backbone LR = 2.5e-5 trong experiment này.

Lý do: lightweight pretrained backbone cần thích nghi khá mạnh từ ImageNet classification sang density/count-mass prediction.

## 8.2. Không weight decay norm/bias

Recommended param grouping:

```python
decay = []
no_decay = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue

    lname = name.lower()
    if (
        p.ndim == 1
        or name.endswith(".bias")
        or "bn" in lname
        or "norm" in lname
    ):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=1e-4,
)
```

Nếu dispersion frozen, criterion không cần param group.

---

# 9. LR schedule cho 2000 epochs

Recommended schedule đơn giản, reproducible:

## 9.1. Warmup

Epoch 0–49:

\[
LR:10^{-5}\rightarrow10^{-4}.
\]

## 9.2. Main schedule

Sau warmup dùng cosine:

\[
LR_t
=
LR_{min}
+
\frac12(LR_{max}-LR_{min})
\left(1+\cos\frac{\pi t}{T}\right)
\]

với:

```yaml
warmup_epochs: 50
lr_max: 1.0e-4
lr_min: 1.0e-6
main_epochs: 1950
```

Pseudo-code:

```python
if epoch < 50:
    alpha = (epoch + 1) / 50.0
    lr = 1e-5 + alpha * (1e-4 - 1e-5)
else:
    t = (epoch - 50) / (2000 - 50)
    lr = 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * t))
```

Không dùng ReduceLROnPlateau cho controlled experiment đầu tiên.

---

# 10. AMP và gradient safety

Keep AMP cho model forward.

NB phải float32 như hiện tại.

Recommended training structure:

```python
optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    d_map = model(images)
    d_degraded = model(images_degraded) if use_robust_batch else None

    # criterion internally casts NB math to fp32
    loss, loss_dict = criterion(
        d_map=d_map,
        gt_block_counts=gt_blocks,
        gt_z_alloc=gt_z_alloc,
        gt_counts=gt_counts,
        d_degraded=d_degraded,
        progress=progress,
    )

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Thêm finite checks:

```python
if not torch.isfinite(loss):
    raise FloatingPointError(f"Non-finite loss: {loss.item()}")
```

---

# 11. Data pipeline — không thay trong Experiment 1, sửa trong Experiment 2

Để biết loss có phải root cause không, **Experiment 1 phải giữ data augmentation giống run hiện tại**.

Sau khi Experiment 1 xong, chạy Experiment 2 với data fixes dưới đây.

---

# 12. Data Fix A — isotropic resize, tuyệt đối không méo aspect ratio

Current bad pattern:

```python
new_w = max(round(w * scale), crop_size)
new_h = max(round(h * scale), crop_size)
```

Có thể tạo scale x/y khác nhau.

Thay bằng:

```python
scale = random.uniform(scale_min, scale_max)

# One common isotropic scale that also guarantees crop feasibility.
scale = max(scale, crop_size / float(w), crop_size / float(h))

new_w = int(round(w * scale))
new_h = int(round(h * scale))

image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
scaled_points = points.copy().astype(np.float32)
scaled_points *= scale
```

### Controlled range

Experiment 2 dùng:

\[
\boxed{scale\in[0.8,1.2]}.
\]

Không dùng `[0.75, 2.0]` trong experiment kiểm chứng đầu tiên.

Sau đó mới ablate wide-scale augmentation.

---

# 13. Data Fix B — safe crop cho large/isolated people

## 13.1. Vấn đề

Nếu GT head center ở ngoài crop nhưng visual body/head vẫn lọt vào crop, target bị biến thành background.

Large near-camera people bị ảnh hưởng nhiều nhất.

## 13.2. Nearest-neighbor scale proxy

Với point \(p_i\):

\[
d_i=\min_{j\neq i}\|p_i-p_j\|_2.
\]

Guard radius:

\[
\boxed{
r_i=clip(0.25d_i,8,40)
}
\]

## 13.3. Candidate crop rejection

Một candidate crop rectangle:

\[
[x_0,x_1)\times[y_0,y_1)
\]

bị reject nếu artificial crop edge đi qua guard region của một GT point.

Pseudo-code:

```python
def crop_boundary_is_safe(points, radii, x0, y0, crop_size):
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    for (x, y), r in zip(points, radii):
        # only care about synthetic crop boundaries inside the real image
        if abs(x - x0) < r:
            return False
        if abs(x - x1) < r:
            return False
        if abs(y - y0) < r:
            return False
        if abs(y - y1) < r:
            return False

    return True
```

Try up to 20 candidate crops. Nếu không tìm được, fallback random crop để tránh deadlock.

**Không reject true image border.** Model cần học người thật đứng sát biên ảnh.

---

# 14. Data Fix C — large/border crop sampling

Sau khi safe crop ổn định:

Recommended mixture:

```text
75% normal random crops
15% point-centered on isolated/large-scale proxy points
10% point-centered near true image border
```

Large proxy:

\[
d_{NN}>48\text{ pixels}
\]

hoặc dataset-adaptive q75/q80 của positive kNN distance.

True border distance:

\[
d_{border}=\min(x,y,W-1-x,H-1-y).
\]

Border point:

\[
d_{border}<48.
\]

Không tăng deploy params.

---

# 15. Predict() border fix bắt buộc

File: `models/hpc_lite.py`

Không dùng:

```python
out_h = h // 4
out_w = w // 4
```

Thay:

```python
out_h = (h + 3) // 4
out_w = (w + 3) // 4
```

hoặc:

```python
out_h = math.ceil(h / 4)
out_w = math.ceil(w / 4)
```

Sau đó:

```python
d_valid = d_padded[..., :out_h, :out_w]
count = d_valid.sum(dim=(-1, -2, -3))
```

### Padding

Reflect padding có thể mirror người ở sát border. Controlled evaluation nên ablate:

```python
F.pad(..., mode="replicate")
```

vs

```python
F.pad(..., mode="constant", value=normalized_zero)
```

Nhưng không đổi padding và loss cùng lúc trong Experiment 1.

---

# 16. Robust second view

Robustness chỉ nên chạy batch-level, không random key per sample.

Recommended:

```python
use_robust_batch = random.random() < 0.30
```

Nếu true:

```python
images_degraded = photometric_batch_transform(images)
```

Không để dataset trả `image_degraded` chỉ cho một subset samples trong cùng batch.

Robust loss weight mới:

\[
0.1\rightarrow\boxed{0.05}.
\]

Lý do: giai đoạn này ưu tiên count accuracy trước.

---

# 17. Training plan 2000 epochs

## Experiment 0 — reproduction sanity

Mục tiêu: verify evaluation pipeline.

- architecture: unchanged
- checkpoint: current best model
- evaluate full val set 3 lần deterministic
- assert identical MAE
- test odd image sizes
- verify no floor crop in `predict()`
- log per-image predictions

Không train.

---

## Experiment 1 — Objective-only

**Không đổi architecture. Không đổi data.**

Thay:

- direct count loss ON;
- HNB 0.35;
- allocation 0.15;
- HN 0.10;
- empty 0.25;
- log-global 0.10;
- robust 0.05;
- dispersion frozen;
- backbone LR 1e-4;
- batch 16;
- 2000 epochs.

Acceptance:

\[
MAE_{new}<MAE_{current}
\]

Strong signal nếu giảm ≥ 3 MAE.

---

## Experiment 1B — direct-count ablation

Same as Experiment 1 nhưng:

```yaml
lambda_count: 0.0
```

Mục đích: chứng minh direct count objective thật sự đóng góp.

Nếu Exp1 tốt hơn Exp1B rõ rệt => hypothesis được xác nhận.

---

## Experiment 1C — dispersion ablation

Same as Exp1 nhưng `learn_dispersion=True`.

So:

```text
fixed r_B
vs
learned r_B
```

Nếu fixed tốt hơn => learned uncertainty đang giảm pressure lên mean count.

---

## Experiment 2 — Data geometry fix

Lấy best objective từ Exp1.

Thêm:

- isotropic scale;
- range [0.8,1.2];
- safe crop;
- optional large/border sampling.

Không sửa architecture.

---

## Experiment 3 — architecture only after objective/data settle

Chỉ khi Exp1/2 đã ổn định.

Không cố giảm xuống 0.17M ngay.

Target budget hợp lý:

\[
\boxed{0.22M-0.28M}
\]

với nguyên tắc:

> compress backbone/channel trước, nhưng giữ đủ task-specific spatial mixer sau compression.

Không dùng depthwise cho mọi spatial mixing layer.

---

# 18. Recommended logging every training step/epoch

## Per step

```text
loss_total
loss_count
loss_hnb
loss_alloc
alloc_entropy
alloc_kl
loss_hn
loss_empty
loss_global
loss_rob
batch_count_mae
mean_pred_count
mean_gt_count
mean_signed_count_error
lr
```

## Per epoch

```text
train_MAE
val_MAE
val_RMSE
val_NAE (dataset-specific)
empty_MAE
dense_top10_MAE
border_point_recall_proxy
large_point_mass_recall_proxy
```

Dispersion:

```text
r_16
r_32
r_64 / r_96
```

Even if frozen, log them.

---

# 19. New diagnostics for large/border failure

For each GT point \(p_i\), compute:

\[
d_{border}(i)=\min(x_i,y_i,W-1-x_i,H-1-y_i)
\]

and nearest-neighbor distance:

\[
d_{NN}(i).
\]

Predicted local mass around point:

\[
m_i(r)=\sum_{(u,v)\in W_i(r)}D_{uv}.
\]

Use image-space radius 32 px => stride-4 map radius 8 cells.

Groups:

```text
border:      d_border < 32
interior:    d_border >= 64
small/dense: d_NN < 16
medium:      16 <= d_NN <= 48
large/sparse:d_NN > 48
```

Report:

```text
mean local mass: border+large
mean local mass: interior+large
mean local mass: border+small
mean local mass: interior+small
```

Nếu `border+large` thấp nhất rõ ràng => data/border hypothesis được xác nhận.

---

# 20. Evaluation correctness

## 20.1. Same number of predictions and GT

Trước MAE:

```python
pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
gt = np.asarray(targets, dtype=np.float64).reshape(-1)

if pred.shape != gt.shape:
    raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
```

Không cho NumPy broadcasting silent.

## 20.2. NWPU NAE

Official-style:

\[
NAE_i=\frac{|\hat C_i-C_i|}{C_i},\quad C_i>0.
\]

Zero-count images excluded from NAE average.

Không dùng:

\[
\frac{|\hat C-C|}{C+1}.
\]

---

# 21. Checkpoint requirements

Checkpoint phải save:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "criterion_state_dict": criterion.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    "best_val_mae": best_val_mae,
    "config": config,
}
```

Dù dispersion frozen, save criterion state để exact resume.

---

# 22. Recommended config — Experiment 1

```yaml
experiment_name: hpc_objective_direct_count_v1

training:
  epochs: 2000
  batch_size: 16
  amp: true
  grad_clip: 1.0
  seed: 42

optimizer:
  type: AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
  no_decay_norm_bias: true

scheduler:
  type: warmup_cosine
  warmup_epochs: 50
  lr_start: 1.0e-5
  lr_max: 1.0e-4
  lr_min: 1.0e-6

loss:
  lambda_count: 1.0
  count_scale: 100.0
  lambda_hnb: 0.35
  lambda_alloc: 0.15
  lambda_hn: 0.10
  lambda_empty: 0.25
  lambda_global: 0.10
  lambda_rob: 0.05
  global_count_mode: log_smooth_l1
  learn_dispersion: false
  enable_curriculum: true

curriculum:
  phase_a_end: 0.05
  phase_b_end: 0.15

robustness:
  batch_probability: 0.30
```

---

# 23. Recommended config — Experiment 2

```yaml
experiment_name: hpc_objective_direct_count_safe_geometry_v1

# inherit all Experiment-1 settings

data:
  scale_min: 0.8
  scale_max: 1.2
  isotropic_resize: true
  safe_crop: true
  safe_crop_max_trials: 20
  safe_crop_guard_factor: 0.25
  safe_crop_guard_min: 8
  safe_crop_guard_max: 40

sampling:
  normal_crop_prob: 0.75
  large_point_crop_prob: 0.15
  border_point_crop_prob: 0.10
```

---

# 24. Unit/regression tests agent phải viết

## Loss tests

1. `DirectCountL1Loss == 0` khi predicted sum = GT.
2. Count error +100 với `count_scale=100` => loss ≈1.
3. Scaling entire map by factor 2 changes direct-count loss đúng như expected.
4. Allocation distribution invariance test giữ nguyên để chứng minh nó không thay count magnitude.
5. Criterion all lambdas zero => total exactly zero at progress 0.01, 0.10, 0.50, 1.0.

## Dispersion tests

6. `learn_dispersion=False` => no grad on raw dispersion.
7. inverse-softplus finite for `1, 10, 100, 1000`.
8. MoM initialization never exceeds configured cap.

## Geometry tests

9. Aspect ratio preserved after resize.
10. Point transform uses exactly same scalar as image.
11. Safe crop rejects synthetic edge crossing point guard.
12. True image border points are still allowed.

## Inference tests

13. 448×448 => 112×112 output.
14. 672×672 => 168×168 output.
15. 449×451 => output uses ceil stride cells, no valid border drop.
16. Count equals exact sum of returned `d_valid`.

## Metric tests

17. prediction/GT length mismatch raises error.
18. NWPU NAE excludes GT=0.

---

# 25. Stop criteria và checkpoint selection

Không select checkpoint bằng training loss.

Select bằng:

\[
\boxed{\text{minimum validation MAE}}
\]

Tie-break:

1. lower RMSE;
2. lower dense-top10 MAE;
3. lower border-large error.

Training 2000 epochs không có nghĩa bắt buộc final epoch là best.

Save:

```text
best_mae.pt
best_rmse.pt
last.pt
```

---

# 26. What NOT to change in Experiment 1

Để giữ experiment interpretable, **không**:

- đổi backbone;
- đổi FPN width;
- thêm scale router;
- thêm SimAM;
- thêm attention;
- đổi output stride;
- đổi allocation target;
- đổi HNB block scales;
- thêm teacher/KD;
- đổi crop geometry cùng lúc.

Experiment 1 chỉ trả lời:

> "Có phải objective/training hiện tại đang không aligned với MAE không?"

---

# 27. Decision tree sau Experiment 1

## Case A — MAE giảm mạnh (>3–5)

Kết luận:

\[
\boxed{objective\ was\ a\ major\ bottleneck}
\]

Tiếp tục Experiment 2 data fixes.

## Case B — MAE chỉ giảm 0–2

Loss có tác dụng nhưng architecture/task mixer vẫn bottleneck.

Sau Experiment 2 mới cân nhắc architecture.

## Case C — MAE xấu hơn

Check:

1. direct count weight quá mạnh;
2. backbone LR quá cao;
3. direct loss gây unstable early training;
4. batch mean dominated by extreme counts.

Ablate:

```yaml
lambda_count: 0.5
count_scale: 200
```

Không bỏ hypothesis ngay sau một run.

---

# 28. Optional robust version của direct count loss

Nếu raw L1 quá aggressive với extreme outliers, dùng Huber trên scaled residual:

\[
e_i=\frac{\hat C_i-C_i}{100}
\]

\[
L_{count}^{Huber}=SmoothL1(e_i,0).
\]

Code:

```python
class DirectCountHuberLoss(nn.Module):
    def __init__(self, count_scale: float = 100.0, beta: float = 1.0):
        super().__init__()
        self.count_scale = float(count_scale)
        self.beta = float(beta)

    def forward(self, d_map, gt_counts):
        pred_counts = d_map.sum(dim=(-1, -2, -3))
        gt_counts = gt_counts.to(pred_counts.device, dtype=pred_counts.dtype)
        residual = (pred_counts - gt_counts) / self.count_scale
        return F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=self.beta,
        )
```

**Không dùng version này trước raw L1.** Raw L1 aligned trực tiếp với benchmark MAE hơn.

---

# 29. Final recommended order of implementation

Agent nên sửa theo đúng thứ tự:

1. `DirectCountL1Loss`.
2. Criterion weights + curriculum factorization.
3. Freeze dispersion + stable inverse softplus.
4. Optimizer LR 1e-4 all trainable model params.
5. Add diagnostics/logging.
6. Fix checkpoint criterion/scaler state.
7. Verify evaluation/predict border handling.
8. Run Experiment 1.
9. Sau kết quả Exp1 mới patch isotropic/safe crop.
10. Sau Exp2 mới xem architecture.

---

# 30. Agent implementation checklist

Before coding:

- [ ] Locate actual files used by the 2000-epoch run; do not patch stale duplicate files.
- [ ] Confirm current model architecture and parameter count.
- [ ] Confirm batch size = 16.
- [ ] Confirm optimizer param groups.
- [ ] Confirm criterion parameters currently enter optimizer or are frozen.

Loss:

- [ ] Add direct count loss.
- [ ] Add `lambda_count` and `count_scale`.
- [ ] Reduce HNB/allocation weights.
- [ ] Curriculum multiplies user lambdas, never overrides them.
- [ ] Freeze NB dispersion in Exp1.
- [ ] Log raw batch MAE.

Training:

- [ ] All model trainable params LR=1e-4.
- [ ] 50 epoch warmup.
- [ ] cosine to 1e-6 by epoch 2000.
- [ ] AMP enabled.
- [ ] grad clip 1.0.
- [ ] save best validation MAE.

Evaluation:

- [ ] `predict()` uses ceil output dimensions.
- [ ] prediction count equals `d_valid.sum()`.
- [ ] prediction/GT arrays same length.
- [ ] official dataset metric conventions preserved.

Data — only Exp2:

- [ ] isotropic scale.
- [ ] [0.8,1.2] controlled range.
- [ ] safe crop.
- [ ] true border sampling.
- [ ] large/isolated head sampling.

---

# 31. Primary hypothesis to test

The key hypothesis is:

\[
\boxed{
\text{HPC-Lite currently optimizes calibrated hierarchical distributions better than raw count MAE.}
}
\]

The revised objective explicitly adds:

\[
\boxed{
\text{constant-strength image-count correction}
}
\]

while preserving HNB and allocation as structural regularizers.

If Experiment 1 improves MAE substantially without changing architecture, then future model design should preserve this objective and optimize parameter allocation around it.

---

# 32. Short version for coding agent

> Keep the current architecture unchanged. Add `DirectCountL1Loss = mean(abs(sum(D)-GT))/100`, weight it 1.0, reduce HNB to 0.35 and allocation to 0.15, freeze NB dispersion, use LR=1e-4 for the entire trainable model, batch=16, 2000 epochs, 50-epoch warmup + cosine to 1e-6, log raw batch MAE and signed count bias, select checkpoint by validation MAE. Do not change crop/architecture in Experiment 1. After that, run a second controlled experiment with isotropic [0.8,1.2] scaling and safe-crop/large-border sampling. Ensure `predict()` uses ceil stride-4 valid dimensions and criterion/checkpoint logic is correct.

