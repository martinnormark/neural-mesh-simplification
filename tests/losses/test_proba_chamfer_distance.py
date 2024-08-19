import torch
import pytest
from losses import ProbabilisticChamferDistanceLoss


@pytest.fixture
def pcd_loss():
    return ProbabilisticChamferDistanceLoss()


def test_pcd_loss_zero_distance(pcd_loss):
    P = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
    Ps = P.clone()
    probs = torch.tensor([0.5, 0.5], dtype=torch.float32)

    loss = pcd_loss(P, Ps, probs)
    assert torch.isclose(
        loss, torch.tensor(0.0)
    ), f"Expected loss to be 0, but got {loss.item()}"


def test_pcd_loss_nonzero_distance(pcd_loss):
    P = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
    Ps = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.float32)
    probs = torch.tensor([0.5, 0.5], dtype=torch.float32)

    loss = pcd_loss(P, Ps, probs)
    assert loss > 0, f"Expected loss to be positive, but got {loss.item()}"
    assert torch.isfinite(loss), f"Expected loss to be finite, but got {loss.item()}"


def test_pcd_loss_empty_input(pcd_loss):
    P = torch.empty((0, 3))
    Ps = torch.empty((0, 3))
    probs = torch.empty(0)

    loss = pcd_loss(P, Ps, probs)
    assert loss == 0, f"Expected loss to be 0 for empty input, but got {loss.item()}"


def test_pcd_loss_different_sizes(pcd_loss):
    P = torch.rand((100, 3))
    Ps = torch.rand((50, 3))
    probs = torch.rand(50)

    loss = pcd_loss(P, Ps, probs)
    assert torch.isfinite(loss), f"Expected loss to be finite, but got {loss.item()}"
    assert loss >= 0, f"Expected non-negative loss, but got {loss.item()}"


def test_pcd_loss_gradient(pcd_loss):
    P = torch.randn((10, 3), requires_grad=True)
    Ps = torch.randn((5, 3), requires_grad=True)
    probs = torch.rand(5, requires_grad=True)

    loss = pcd_loss(P, Ps, probs)
    loss.backward()

    assert P.grad is not None, "Gradient for P should not be None"
    assert Ps.grad is not None, "Gradient for Ps should not be None"
    assert probs.grad is not None, "Gradient for probabilities should not be None"
    assert torch.isfinite(P.grad).all(), "Gradient for P should be finite"
    assert torch.isfinite(Ps.grad).all(), "Gradient for Ps should be finite"
    assert torch.isfinite(
        probs.grad
    ).all(), "Gradient for probabilities should be finite"
