import pytest
import torch

from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.hpc_lite import HPCLite, inv_softplus


def test_inv_softplus():
    """inv_softplus must be exact inverse of softplus."""
    values = [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0]
    for v in values:
        b = inv_softplus(v)
        rec = float(torch.nn.functional.softplus(torch.tensor(b)))
        assert abs(rec - v) / v < 1e-4


def test_one_image_overfit():
    """HPCLite model with NTPCLoss (R4) must overfit a single synthetic crop to |N_hat - N| < 0.5."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = HPCLite(pretrained=False, neck_width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    # 1 synthetic image with N = 5 points
    img = torch.randn(1, 3, 256, 256, device=device)
    pts = torch.tensor([[50.0, 50.0], [100.0, 120.0], [150.0, 80.0], [200.0, 200.0], [220.0, 30.0]])
    gt_N = float(len(pts))  # 5.0
    tree = build_exact_count_pyramid([pts], height=256, width=256)
    targets = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"].to(device)

    criterion = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16")).to(device)

    # Record initial prediction
    model.eval()
    with torch.no_grad():
        initial_mass = model(img)
        initial_pred_count = float(initial_mass.sum().item())
        initial_count_error = abs(initial_pred_count - gt_N)

    model.train()
    losses = []
    for step in range(80):
        optimizer.zero_grad()
        mass = model(img)
        loss, _ = criterion(mass, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        final_mass = model(img)
        final_pred_count = float(final_mass.sum().item())
        final_count_error = abs(final_pred_count - gt_N)

    # Strict overfit assertions
    assert losses[-1] < losses[0] * 0.40, f"Loss did not decrease significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert final_count_error < initial_count_error, (
        f"Count error did not improve: initial={initial_count_error:.3f}, final={final_count_error:.3f}"
    )
    assert final_count_error < 0.5, (
        f"Final count error |N_hat - N| = {final_count_error:.3f} >= 0.5 (pred={final_pred_count:.3f}, gt={gt_N})"
    )


@pytest.mark.slow
def test_ten_images_overfit():
    """HPCLite model overfit sanity on 10 synthetic crops."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = HPCLite(pretrained=False, neck_width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    images = torch.randn(10, 3, 256, 256, device=device)
    points_list = [torch.rand(torch.randint(2, 20, (1,)).item(), 2) * 250.0 for _ in range(10)]
    tree = build_exact_count_pyramid(points_list, height=256, width=256)
    targets = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"].to(device)

    criterion = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16")).to(device)

    model.train()
    losses = []
    for step in range(15):
        optimizer.zero_grad()
        mass = model(images)
        loss, _ = criterion(mass, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
