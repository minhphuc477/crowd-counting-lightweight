"""
HPC-S-SR48 full regression test suite.
Tests every layer of the stack: architecture, data pipeline, losses, train schema.
Run:  python tests/test_sr48.py
All tests must pass before starting any training run.
"""
import sys, math, traceback
import numpy as np
import torch
import torch.nn as nn

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []

def check(name, cond, detail=""):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}" + (f" — {detail}" if detail else ""))
        _failures.append(name)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────
# 1. BLOCKS
# ─────────────────────────────────────────────────────────────────
section("1. Blocks (ConvGNAct, DSResidual, MultiPoolContext, SimAM)")
try:
    from hpc.models.blocks import ConvGNAct, DSResidual, MultiPoolContext, SimAM, make_group_norm

    # ConvGNAct
    m = ConvGNAct(32, 48, kernel_size=1)
    x = torch.randn(2, 32, 56, 56)
    y = m(x)
    check("ConvGNAct output shape", y.shape == (2, 48, 56, 56), y.shape)

    # DSResidual (depthwise + pointwise, residual)
    dr = DSResidual(48)
    y2 = dr(y)
    check("DSResidual output shape", y2.shape == (2, 48, 56, 56), y2.shape)

    # MultiPoolContext: zero learnable params
    mpc = MultiPoolContext()
    p = sum(p.numel() for p in mpc.parameters())
    check("MultiPoolContext params == 0", p == 0, f"got {p}")
    y3 = mpc(y2)
    check("MultiPoolContext output shape", y3.shape == (2, 48, 56, 56), y3.shape)

    # SimAM: zero learnable params
    sa = SimAM()
    p2 = sum(p.numel() for p in sa.parameters())
    check("SimAM params == 0", p2 == 0, f"got {p2}")
    y4 = sa(y3)
    check("SimAM output shape matches input", y4.shape == y3.shape, y4.shape)

    # make_group_norm
    gn = make_group_norm(48)
    check("make_group_norm returns GroupNorm", isinstance(gn, nn.GroupNorm))

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("blocks")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 2. BACKBONE
# ─────────────────────────────────────────────────────────────────
section("2. ShuffleNetV2PyramidBackbone")
try:
    from hpc.models.backbone import ShuffleNetV2PyramidBackbone
    bb = ShuffleNetV2PyramidBackbone(pretrained=False)
    p = sum(x.numel() for x in bb.parameters())
    check(f"Backbone params == 143136", p == 143136, f"got {p}")
    check("out_channels == [24,48,96,192]", bb.out_channels == [24, 48, 96, 192])

    bb.eval()
    with torch.no_grad():
        c4, c8, c16, c32 = bb(torch.randn(1, 3, 448, 448))
    check("c4 channels == 24",  c4.shape[1] == 24,  c4.shape)
    check("c8 channels == 48",  c8.shape[1] == 48,  c8.shape)
    check("c16 channels == 96", c16.shape[1] == 96, c16.shape)
    check("c32 channels == 192",c32.shape[1] == 192,c32.shape)
    check("c4 spatial == /4",   c4.shape[-2:] == torch.Size([112,112]), c4.shape)
    check("c8 spatial == /8",   c8.shape[-2:] == torch.Size([56,56]),   c8.shape)
    check("c16 spatial == /16", c16.shape[-2:] == torch.Size([28,28]),  c16.shape)
    check("c32 spatial == /32", c32.shape[-2:] == torch.Size([14,14]),  c32.shape)

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("backbone")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 3. NECK (SSER)
# ─────────────────────────────────────────────────────────────────
section("3. ScaleRoutedFusionNeck (SSER)")
try:
    from hpc.models.neck import ScaleRoutedFusionNeck
    neck = ScaleRoutedFusionNeck(in_channels=[24,48,96,192], width=48)

    # Router param count
    scorer_p = sum(p.numel() for p in neck.shared_scorer.parameters())
    bias_p   = neck.scale_bias.numel()
    check("shared_scorer params == 48", scorer_p == 48, f"got {scorer_p}")
    check("scale_bias params == 4",     bias_p   == 4,  f"got {bias_p}")
    check("Router total == 52", scorer_p + bias_p == 52, f"got {scorer_p+bias_p}")

    # Zero init → initial route is exactly uniform
    with torch.no_grad():
        assert torch.all(neck.shared_scorer.weight == 0), "shared_scorer must be zero-init"
        assert torch.all(neck.scale_bias == 0), "scale_bias must be zero-init"
    check("Router zero-init (uniform start)", True)

    # Forward shapes
    neck.eval()
    c4  = torch.randn(1, 24, 112, 112)
    c8  = torch.randn(1, 48, 56, 56)
    c16 = torch.randn(1, 96, 28, 28)
    c32 = torch.randn(1, 192, 14, 14)
    with torch.no_grad():
        fused = neck(c4, c8, c16, c32)
    check("Neck output shape (B,48,112,112)", fused.shape == (1,48,112,112), fused.shape)

    # return_routes=True
    with torch.no_grad():
        fused2, route_info = neck(c4, c8, c16, c32, return_routes=True)
    routes8 = route_info["routes8"]
    routes4 = route_info["routes4"]
    check("routes8 shape (1,4,56,56)",   routes8.shape == (1,4,56,56), routes8.shape)
    check("routes4 shape (1,4,112,112)", routes4.shape == (1,4,112,112), routes4.shape)

    # routes8 sum-to-1 per location
    r_sum = routes8.sum(dim=1)  # (1,56,56) all ones
    check("routes8 sum-to-1", torch.allclose(r_sum, torch.ones_like(r_sum), atol=1e-5),
          f"max dev={( r_sum-1).abs().max().item():.2e}")

    # Uniform start: all routes == 0.25 with zero-init scorer
    check("Initial routes == 0.25",
          torch.allclose(routes8, torch.full_like(routes8, 0.25), atol=1e-5),
          f"mean={routes8.mean():.4f}")

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("neck")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 4. HPCLiteSR48 full model
# ─────────────────────────────────────────────────────────────────
section("4. HPCLiteSR48 full model")
try:
    from hpc.models.hpc_lite import HPCLiteSR48
    model = HPCLiteSR48(pretrained=False)

    total_p = sum(p.numel() for p in model.parameters())
    check(f"Deploy params == 173909", total_p == 173909, f"got {total_p}")

    model.eval()
    with torch.no_grad():
        # Standard input
        d = model(torch.randn(2, 3, 448, 448))
        check("448x448 output shape (2,1,112,112)", d.shape == (2,1,112,112), d.shape)

        # 672x672
        d2 = model(torch.randn(1, 3, 672, 672))
        check("672x672 output shape (1,1,168,168)", d2.shape == (1,1,168,168), d2.shape)

        # Strictly positive + finite
        check("Output strictly positive", bool((d > 0).all()), f"min={d.min():.4e}")
        check("Output finite", bool(d.isfinite().all()))

        # return_aux
        d3, aux = model(torch.randn(1, 3, 448, 448), return_aux=True)
        check("return_aux returns (d, dict)", isinstance(aux, dict))
        check("aux has routes8 key", "routes8" in aux)
        check("aux has routes4 key", "routes4" in aux)
        routes8 = aux["routes8"]
        check("routes8 shape (1,4,56,56)", routes8.shape == (1,4,56,56), routes8.shape)
        rs = routes8.sum(dim=1)
        check("routes8 sum-to-1", torch.allclose(rs, torch.ones_like(rs), atol=1e-5))

        # predict() with pad_multiple=32
        cnt, dv = model.predict(torch.randn(1, 3, 449, 451), pad_multiple=32)
        exp_h, exp_w = math.ceil(449/4), math.ceil(451/4)
        check(f"predict 449x451 output {exp_h}x{exp_w}",
              dv.shape[-2] == exp_h and dv.shape[-1] == exp_w, dv.shape)
        check("predict count == sum(d_valid)", abs(cnt.item() - dv.sum().item()) < 1e-3,
              f"cnt={cnt.item():.4f} vs sum={dv.sum().item():.4f}")

        # Odd multiples
        cnt2, dv2 = model.predict(torch.randn(1, 3, 320, 320), pad_multiple=32)
        check("predict 320x320 output 80x80", dv2.shape[-2:] == torch.Size([80,80]), dv2.shape)

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("model")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 5. ROUTING TARGET BUILDER
# ─────────────────────────────────────────────────────────────────
section("5. build_routing_target")
try:
    from hpc.targets.routing_target import build_routing_target

    # 3 points with different density regimes
    pts = np.array([[50, 50], [250, 250], [400, 400]], dtype=np.float32)
    dnn = np.array([10.0, 35.0, 80.0], dtype=np.float32)

    res = build_routing_target(pts, dnn, 448, 448, route_stride=8)
    q, mask = res["gt_route_q"], res["gt_route_mask"]

    check("q shape (4,56,56)", q.shape == (4,56,56), q.shape)
    check("mask shape (56,56)", mask.shape == (56,56), mask.shape)
    check("mask has True entries", bool(mask.any()))

    # Supervised cells sum to 1
    q_sup  = q[:, mask]
    sums   = q_sup.sum(dim=0)
    check("Supervised q sum-to-1",
          bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5)),
          f"max dev={( sums-1).abs().max().item():.2e}")

    # Dense point (d_nn=10, gx=6, gy=6) → prefers /4 (index 0)
    gx0, gy0 = int(50/8), int(50/8)
    d0 = q[:, gy0, gx0]
    check("Dense point prefers /4 (α0 largest)", float(d0[0]) > float(d0[1]) > float(d0[2]),
          d0.tolist())

    # Large/isolated point (d_nn=80) → prefers /32 (index 3)
    gx2, gy2 = int(400/8), int(400/8)
    d2 = q[:, gy2, gx2]
    check("Large point prefers /32 (α3 largest)", float(d2[3]) > float(d2[2]) > float(d2[0]),
          d2.tolist())

    # Empty image → no supervised cells, q is uniform
    res_empty = build_routing_target(
        np.empty((0,2), dtype=np.float32), np.empty(0, dtype=np.float32), 448, 448
    )
    check("Empty image: mask all False", not bool(res_empty["gt_route_mask"].any()))
    q_unif = res_empty["gt_route_q"]
    check("Empty image: q uniform 0.25",
          bool(torch.allclose(q_unif, torch.full_like(q_unif, 0.25), atol=1e-6)))

    # Single-point image
    res_single = build_routing_target(
        np.array([[100,100]], dtype=np.float32), np.array([5.0], dtype=np.float32), 448, 448
    )
    check("Single point: mask has 1 cell", int(res_single["gt_route_mask"].sum()) == 1)

    # Multiple points in same cell → averaged distribution
    pts2 = np.array([[8,8],[10,10]], dtype=np.float32)   # both in cell (1,1)
    dnn2 = np.array([10.0, 10.0], dtype=np.float32)      # same regime
    res2 = build_routing_target(pts2, dnn2, 448, 448)
    check("Same cell two points: mask has 1 cell", int(res2["gt_route_mask"].sum()) == 1)

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("routing_target")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 6. ROUTING SUPERVISION LOSS
# ─────────────────────────────────────────────────────────────────
section("6. RoutingSupervisionLoss — KL(q ‖ α)")
try:
    from hpc.losses.routing import RoutingSupervisionLoss

    loss_fn = RoutingSupervisionLoss()
    B, H8, W8 = 2, 56, 56

    routes = torch.softmax(torch.randn(B, 4, H8, W8), dim=1)
    from hpc.targets.routing_target import build_routing_target
    res = build_routing_target(
        np.array([[100,100],[300,300]], dtype=np.float32),
        np.array([10.0, 80.0], dtype=np.float32),
        448, 448
    )
    q_b    = res["gt_route_q"].unsqueeze(0).expand(B,-1,-1,-1)
    mask_b = res["gt_route_mask"].unsqueeze(0).expand(B,-1,-1)

    # Random routes → positive KL
    l_rand = loss_fn(routes, q_b, mask_b)
    check("KL loss is finite", bool(l_rand.isfinite()), f"{l_rand.item()}")
    check("KL loss >= 0", float(l_rand.item()) >= 0, f"{l_rand.item()}")
    check("KL loss > 0 for non-perfect routes", float(l_rand.item()) > 0)

    # Perfect routing (alpha == q) → KL == 0
    l_perf = loss_fn(q_b, q_b, mask_b)
    check("KL(q‖q) ≈ 0 (perfect routes)", float(l_perf.item()) < 1e-5,
          f"got {l_perf.item():.2e}")

    # Empty supervision mask → zero loss (no grad)
    empty_mask = torch.zeros(B, H8, W8, dtype=torch.bool)
    l_empty = loss_fn(routes, q_b, empty_mask)
    check("KL loss == 0 for empty mask", float(l_empty.item()) == 0.0)

    # Gradient flows back to routes
    routes_g = torch.softmax(torch.randn(1, 4, H8, W8, requires_grad=False), dim=1)
    routes_g = routes_g.detach().requires_grad_(True)
    l_g = loss_fn(routes_g, q_b[:1], mask_b[:1])
    l_g.backward()
    check("Gradient flows to routes8", routes_g.grad is not None)
    check("Gradient is finite", bool(routes_g.grad.isfinite().all()))

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("routing_loss")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 7. FULL CRITERION (all terms, curriculum schedule)
# ─────────────────────────────────────────────────────────────────
section("7. HPCLossCriterion — 10 loss terms with curriculum")
try:
    from hpc.losses.criterion import HPCLossCriterion
    from hpc.models.hpc_lite import HPCLiteSR48
    from hpc.targets.routing_target import build_routing_target

    model = HPCLiteSR48(pretrained=False)
    model.eval()

    criterion = HPCLossCriterion(
        block_sizes=[16, 32, 64],
        allocation_block=16,
        lambda_hnb=1.0,
        lambda_alloc=0.5,
        lambda_hn=0.25,
        lambda_empty=0.5,
        lambda_global=0.5,
        lambda_direct=0.5,
        lambda_special=0.25,
        lambda_rob=0.1,
        lambda_kd=0.0,
        lambda_route=0.1,
    )

    res = build_routing_target(
        np.array([[100,100],[300,300]], dtype=np.float32),
        np.array([10.0, 80.0], dtype=np.float32),
        448, 448
    )
    gt_route_q    = res["gt_route_q"].unsqueeze(0)    # (1,4,56,56)
    gt_route_mask = res["gt_route_mask"].unsqueeze(0) # (1,56,56)

    with torch.no_grad():
        d, aux = model(torch.randn(1,3,448,448), return_aux=True)
    routes8 = aux["routes8"]

    gt_blocks = {b: torch.zeros(1, 448//b, 448//b) for b in [16,32,64]}
    gt_z      = torch.zeros(1, 112, 112)
    gt_cnt    = torch.zeros(1)

    for prog, label in [(0.05, "phase0 (0-10%)"), (0.20, "phase1 (10-30%)"), (0.55, "phase2 (30%+)")]:
        total, ld = criterion(
            d, gt_blocks, gt_z, gt_cnt,
            routes8=routes8, gt_route_q=gt_route_q, gt_route_mask=gt_route_mask,
            progress=prog,
        )
        check(f"Total loss finite [{label}]", bool(total.isfinite()), f"{total.item():.4f}")
        check(f"loss_route in loss_dict [{label}]", "loss_route" in ld)
        check(f"weight_route in loss_dict [{label}]", "weight_route" in ld)
        check(f"loss_total in loss_dict [{label}]", "loss_total" in ld)

    # Curriculum: route weight = 0 in phase0, > 0 after
    _, ld0 = criterion(d, gt_blocks, gt_z, gt_cnt,
                       routes8=routes8, gt_route_q=gt_route_q, gt_route_mask=gt_route_mask,
                       progress=0.05)
    _, ld1 = criterion(d, gt_blocks, gt_z, gt_cnt,
                       routes8=routes8, gt_route_q=gt_route_q, gt_route_mask=gt_route_mask,
                       progress=0.55)
    check("Route weight == 0 in phase0", float(ld0["weight_route"].item()) == 0.0)
    check("Route weight == 0.1 in phase2", abs(float(ld1["weight_route"].item()) - 0.1) < 1e-6)

    # No routing args → loss_route == 0
    _, ld_no_r = criterion(d, gt_blocks, gt_z, gt_cnt, progress=0.55)
    check("loss_route == 0 when no routing args",
          float(ld_no_r["loss_route"].item()) == 0.0)

    # Expected loss_dict keys
    expected_keys = [
        "loss_hnb", "loss_alloc", "loss_hn", "loss_empty",
        "loss_global", "loss_direct", "loss_special", "loss_rob",
        "loss_kd", "loss_route", "loss_total",
    ]
    for k in expected_keys:
        check(f"loss_dict has key '{k}'", k in ld1)

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("criterion")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 8. DATA TRANSFORMS — point_dnn returned
# ─────────────────────────────────────────────────────────────────
section("8. ScaleAwareSafeGeometricTransforms — point_dnn output")
try:
    from hpc.data.transforms import ScaleAwareSafeGeometricTransforms
    from PIL import Image as PILImage

    tf = ScaleAwareSafeGeometricTransforms(crop_size=448, scale_range=(0.75, 2.0))
    img = PILImage.fromarray(np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8))
    pts = np.random.uniform(50, 700, (30, 2)).astype(np.float32)

    out = tf(img, pts)
    check("Transform returns dict", isinstance(out, dict))
    check("'point_dnn' key present", "point_dnn" in out)
    check("'points' key present", "points" in out)
    check("'point_large_flags' key present", "point_large_flags" in out)
    check("'point_true_border_flags' key present", "point_true_border_flags" in out)

    dnn_out = out["point_dnn"]
    pts_out = out["points"]
    check("point_dnn length == len(surviving points)",
          len(dnn_out) == len(pts_out),
          f"dnn={len(dnn_out)} pts={len(pts_out)}")
    if len(dnn_out) > 0:
        check("point_dnn all > 0", bool((dnn_out > 0).all()),
              f"min={dnn_out.min():.2f}")
        check("point_dnn dtype float32",
              str(dnn_out.dtype) in ("float32", "<class 'numpy.float32'>"))

    # Empty image (0 points) should not crash
    out_empty = tf(img, np.empty((0,2), dtype=np.float32))
    check("Empty image: no crash", True)
    check("Empty image: point_dnn length 0", len(out_empty["point_dnn"]) == 0)

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("transforms")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# 9. GRAD FLOW through full model + criterion
# ─────────────────────────────────────────────────────────────────
section("9. End-to-end gradient flow")
try:
    from hpc.models.hpc_lite import HPCLiteSR48
    from hpc.losses.criterion import HPCLossCriterion
    from hpc.targets.routing_target import build_routing_target

    model = HPCLiteSR48(pretrained=False)
    model.train()
    criterion = HPCLossCriterion(
        block_sizes=[16,32,64], allocation_block=16, lambda_route=0.1
    )

    d, aux = model(torch.randn(1,3,448,448), return_aux=True)
    routes8 = aux["routes8"]

    res = build_routing_target(
        np.array([[100,100]], dtype=np.float32),
        np.array([10.0], dtype=np.float32),
        448, 448
    )

    total, _ = criterion(
        d,
        {b: torch.zeros(1, 448//b, 448//b) for b in [16,32,64]},
        torch.zeros(1,112,112),
        torch.zeros(1),
        routes8=routes8,
        gt_route_q=res["gt_route_q"].unsqueeze(0),
        gt_route_mask=res["gt_route_mask"].unsqueeze(0),
        progress=0.55,
    )
    total.backward()

    # Check backbone, neck (including router), head all got gradients
    bb_grad  = any(p.grad is not None for p in model.backbone.parameters())
    neck_grad = any(p.grad is not None for p in model.neck.parameters())
    scorer_g = model.neck.shared_scorer.weight.grad
    bias_g   = model.neck.scale_bias.grad

    check("Backbone gets gradient", bb_grad)
    check("Neck gets gradient", neck_grad)
    check("shared_scorer.weight gets gradient", scorer_g is not None)
    check("scale_bias gets gradient", bias_g is not None)
    if scorer_g is not None:
        check("shared_scorer gradient is finite", bool(scorer_g.isfinite().all()))
    if bias_g is not None:
        check("scale_bias gradient is finite", bool(bias_g.isfinite().all()))

except Exception as e:
    print(f"  [{FAIL}] EXCEPTION: {e}"); _failures.append("grad_flow")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if not _failures:
    print(f"\033[32m  ALL TESTS PASSED ✓\033[0m")
    sys.exit(0)
else:
    print(f"\033[31m  {len(_failures)} TEST(S) FAILED:\033[0m")
    for f in _failures:
        print(f"    - {f}")
    sys.exit(1)
