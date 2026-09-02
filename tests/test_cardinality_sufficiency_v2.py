import torch

from hpc.diagnostics.cardinality_sufficiency_v2 import (
    PCAProjector,
    avgpool2x,
    blurpool2x,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_ridge,
    pack_2x2_features,
    pack_child_counts,
    summarize_prediction,
)


def test_pack_geometry_and_order():
    x = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    packed = pack_2x2_features(x)
    assert packed.shape == (1, 2, 2, 4)
    assert torch.equal(packed[0, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))

    y = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    child = pack_child_counts(y)
    assert torch.equal(child[0, 0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))


def test_lossless_pack_contains_all_values():
    x = torch.randn(2, 3, 8, 10)
    packed = pack_2x2_features(x)
    assert packed.numel() == x.numel()
    assert torch.allclose(
        torch.sort(packed.reshape(-1)).values,
        torch.sort(x.reshape(-1)).values,
    )


def test_pool_controls_have_expected_shape():
    x = torch.randn(2, 5, 8, 10)
    assert avgpool2x(x).shape == (2, 5, 4, 5)
    assert blurpool2x(x).shape == (2, 5, 4, 5)


def test_representation_grid_alignment():
    pre = torch.randn(1, 6, 16, 16)
    post = torch.randn(1, 10, 8, 8)
    grids = build_representation_grid(pre, post)
    assert grids["pre_pack"].shape == (1, 8, 8, 24)
    assert grids["native_post"].shape == (1, 8, 8, 10)
    assert grids["avgpool"].shape == (1, 8, 8, 6)
    assert grids["blurpool"].shape == (1, 8, 8, 6)


def test_pca_budget_dimension():
    x = torch.randn(100, 12)
    pca = PCAProjector.fit(x, output_dim=7)
    assert pca.transform(x).shape == (100, 7)
    pca_expand = PCAProjector.fit(x, output_dim=15)
    z = pca_expand.transform(x)
    assert z.shape == (100, 15)
    assert torch.count_nonzero(z[:, 12:]) == 0


def test_ridge_recovers_linear_child_counts():
    torch.manual_seed(0)
    x_train = torch.randn(400, 20)
    w = torch.randn(20, 4)
    y_train = x_train @ w + 0.01 * torch.randn(400, 4)
    x_val = torch.randn(100, 20)
    y_val = x_val @ w
    pred = fit_predict_ridge(x_train, y_train, x_val, alpha=1e-3)
    assert (pred - y_val).abs().mean() < 0.02


def test_summary_has_dense_bins():
    y = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [2, 1, 1, 0], [2, 2, 1, 1]], dtype=torch.float32
    )
    pred = y.clone()
    m = summarize_prediction(pred, y)
    assert m["child_mae"] == 0.0
    assert m["n2p_cells"] == 3
    assert m["n5p_cells"] == 1


def test_bootstrap_diff_sign():
    a = torch.tensor([2.0, 2.0, 4.0, 4.0])
    b = torch.tensor([1.0, 1.0, 2.0, 2.0])
    ids = torch.tensor([0, 0, 1, 1])
    out = bootstrap_image_mean_difference(a, b, ids, n_boot=200, seed=1)
    assert out["mean_diff"] > 0


def test_tiny_mlp_probe_runs():
    from hpc.diagnostics.cardinality_sufficiency_v2 import fit_predict_mlp

    torch.manual_seed(3)
    x = torch.randn(120, 6)
    y = torch.relu(x[:, :4])
    pred = fit_predict_mlp(
        x[:100], y[:100], x[100:], hidden=16, epochs=3, batch_size=32, seed=3
    )
    assert pred.shape == (20, 4)
    assert torch.isfinite(pred).all()
