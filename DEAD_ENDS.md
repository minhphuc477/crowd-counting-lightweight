# DEAD_ENDS.md — Rejected & Banned Directions (Do Not Re-Propose)

Each entry lists a permanently closed path, the verified failure mode, and the safe alternative.

| Anti-Pattern / Rejected Idea | Verified Failure Evidence | Safe Path Instead |
| :--- | :--- | :--- |
| **`selective_gate` (`sfi_use_selective_gate=True`)** | Flatlined at `0.471–0.474` on SHA even under `complementary` mode (`66.64` MAE vs `64.05` without it) due to `GroupNorm` pinning variance. | Use direct parameter-lean `LFI-Lite` (`gamma_hf`) or Stride-2 `C2` fusion without `GroupNorm`-pinned spatial gates. |
| **Unrolled SIRT Solver / `MCBPRO` / `region_head` at inference** | Added heavy latency and ONNX `regions.py` bloat while direct fast-path (`enable_solver=False`) wins on accuracy and hits `38 FPS` CPU ORT (`512`). | Single-pass direct Stride-2 prediction (`enable_solver=False, enable_regional_head=False`). |
| **`dynamic_perspective_windows` (`DPW`), `use_park`, `use_coord_attn`, `use_nesterov`, `use_alternating_bb`, `foreground_gate`, `tdsg`** | Permanently banned in `validator.py`: crop-coordinate aliasing, gradient suppression, kinetic overshoot, or variance explosion. | Pure local/pyramid features; `DSIMR` soft support gate `sigma(z_s)`. |
| **`use_aspp_gap=True` (Global Average Pooling on `L16`)** | Spatially dilutes local crowd density peaks across background and exceeds sub-100k budget (`102k+`). | Sequential gridding-free HDC `(1, 2, 3)` on `L16` with `use_aspp_gap=False`. |
| **`Frankenstein` FFT neck (`aten::fft_rfft2`)** | Fails ONNX opset 17 export (`UnsupportedOperatorError`) and runs `2.2x` slower on CUDA @ 1024 (`193.8` vs `425.5 FPS`). | Pure spatial conv/pooling ops (Laplacian `3x3 avg_pool2d`) 100% compatible with ONNX/TensorRT. |
| **Bounded `torch.tanh` on HF residual (`LFI-Lite`)** | Clamps high-frequency crowd impulses to `[-1, +1]`, capping dense response in `3000+` crowds. | `sfi_hf_act_mode: linear` + `pad_aware_laplacian: true` (`count_include_pad=False`) + `gamma_hf` WD exemption. |
| **Pure `count_loss_mode: nb` (`r=50`) alone** | `1/y` gradient decay makes `y=500` count gradients `7x` weaker per missing person than `y=20`, starving `3000+` scenes. | `count_loss_mode: anscombe` + `lambda_count_l1: 0.02` (`D1`/`D2`). |
| **`GroupNorm` on Stride-2 (`H/2 x W/2`) layers** | Tripled 1-thread CPU ORT latency (`76.65 ms / 13.1 FPS` @ 512, `720.37 ms / 1.4 FPS` @ 1080p) due to CPU memory bandwidth. | Foldable `BatchNorm2d` on Stride-2 layers (`38.53 ms / 25.9 FPS` @ 512, `2.7 FPS` @ 1080p). |
| **`sfi_v2_matrix` (`lr=1e-4, blr=0.1`)** | Silent `7.5x` lower backbone LR than canonical (`lr=3e-4, blr=0.25`); invalidated runs deleted. | Enforce invariant `lr=0.0003, backbone_lr_scale=0.25` across comparable runs. |
