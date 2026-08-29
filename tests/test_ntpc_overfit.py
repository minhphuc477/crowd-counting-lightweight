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
    """HPCLite model with NTPCLoss (R4) must overfit a single synthetic crop in 25 steps."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = HPCLite(pretrained=False, neck_width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # 1 synthetic image with 5 points
    img = torch.randn(1, 3, 256, 256, device=device)
    pts = torch.tensor([[50.0, 50.0], [100.0, 120.0], [150.0, 80.0], [200.0, 200.0], [220.0, 30.0]])
    tree = build_exact_count_pyramid([pts], height=256, width=256)
    targets = {k: tree[k].to(device) for k in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"].to(device)

    criterion = NTPCLoss(NTPCConfig(mode="r4_dtm_tree16")).to(device)

    model.train()
    losses = []
    for step in range(25):
        optimizer.zero_grad()
        mass = model(img)
        loss, _ = criterion(mass, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert losses[-1] < losses[0] * 0.7, f"Loss did not decrease significantly: {losses[0]} -> {losses[-1]}"


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
