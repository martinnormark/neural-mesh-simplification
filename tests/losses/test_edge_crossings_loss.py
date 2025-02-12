import pytest
import torch

from neural_mesh_simplification.losses.edge_crossing_loss import EdgeCrossingLoss

k_val = 2

@pytest.fixture
def loss_fn():
    return EdgeCrossingLoss(k=k_val)


@pytest.fixture
def sample_data():
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.float,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]], dtype=torch.long)
    face_probs = torch.tensor([0.8, 0.6, 0.7, 0.9], dtype=torch.float)
    return vertices, faces, face_probs


def test_find_nearest_triangles(loss_fn):
    vertices = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]],
        dtype=torch.float,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 4, 5]], dtype=torch.long)

    nearest = loss_fn.find_nearest_triangles(vertices, faces)
    assert nearest.shape[0] == faces.shape[0]
    assert nearest.shape[1] == k_val


def test_detect_edge_crossings(loss_fn, sample_data):
    vertices, faces, _ = sample_data

    nearest_triangles = torch.tensor([[1], [0], [3], [2]], dtype=torch.long)
    crossings = loss_fn.detect_edge_crossings(vertices, faces, nearest_triangles)

    # Test expected number of crossings
    # Here, you should check for specific cases based on the vertex configuration
    # Example: You expect 0 crossings if triangles are separate
    assert torch.all(crossings == 0)  # Modify this based on your actual expectations


def test_calculate_loss(loss_fn, sample_data):
    _, _, face_probs = sample_data

    crossings = torch.tensor([1.0, 0.0, 2.0, 1.0], dtype=torch.float)
    loss = loss_fn.calculate_loss(crossings, face_probs)

    num_faces = face_probs.shape[0]
    expected_loss = torch.sum(face_probs * crossings, dtype=torch.float32) / num_faces
    assert torch.isclose(
        loss, torch.tensor(expected_loss)
    ), f"Expected loss: {expected_loss}, but got {loss.item()}"


def test_edge_crossing_loss_full(loss_fn, sample_data):
    vertices, faces, face_probs = sample_data

    # Run the forward pass of the loss function
    loss = loss_fn(vertices, faces, face_probs)

    # Check that the loss value is as expected
    # Ensure expected_loss is a floating-point tensor
    expected_loss = torch.tensor(
        0.0, dtype=torch.float
    )  # Modify this based on your setup
    assert torch.isclose(
        loss, expected_loss
    ), f"Expected loss: {expected_loss.item()}, but got {loss.item()}"
