import torch
import pytest
from losses import OverlappingTrianglesLoss


@pytest.fixture
def loss_fn():
    return OverlappingTrianglesLoss(num_samples=5, k=3)


@pytest.fixture
def sample_data():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float,
    )
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]], dtype=torch.long)
    return vertices, faces


def test_sample_points_from_triangles(loss_fn, sample_data):
    vertices, faces = sample_data

    sampled_points = loss_fn.sample_points_from_triangles(vertices, faces)

    # Check the shape of the sampled points
    expected_shape = (faces.shape[0] * loss_fn.num_samples, 3)
    assert (
        sampled_points.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {sampled_points.shape}"

    # Check that points lie within the bounding box of the mesh
    assert torch.all(
        sampled_points >= vertices.min(dim=0).values
    ), "Sampled points are outside the mesh bounds"
    assert torch.all(
        sampled_points <= vertices.max(dim=0).values
    ), "Sampled points are outside the mesh bounds"


def test_find_nearest_triangles(loss_fn, sample_data):
    vertices, faces = sample_data

    sampled_points = loss_fn.sample_points_from_triangles(vertices, faces)
    nearest_triangles = loss_fn.find_nearest_triangles(sampled_points, vertices, faces)

    # Check the shape of the nearest triangles tensor
    expected_shape = (sampled_points.shape[0], loss_fn.k)
    assert (
        nearest_triangles.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {nearest_triangles.shape}"

    # Check that the indices are within the range of faces
    assert torch.all(nearest_triangles >= 0) and torch.all(
        nearest_triangles < faces.shape[0]
    ), "Invalid triangle indices"


def test_calculate_overlap_loss(loss_fn, sample_data):
    vertices, faces = sample_data

    sampled_points = loss_fn.sample_points_from_triangles(vertices, faces)
    nearest_triangles = loss_fn.find_nearest_triangles(sampled_points, vertices, faces)
    overlap_loss = loss_fn.calculate_overlap_loss(
        sampled_points, vertices, faces, nearest_triangles
    )

    # Check that the overlap loss is a scalar
    assert (
        isinstance(overlap_loss, torch.Tensor) and overlap_loss.dim() == 0
    ), "Overlap loss should be a scalar"

    # For this simple test case, the overlap should be minimal or zero
    assert overlap_loss.item() >= 0, "Overlap loss should be non-negative"


def test_overlapping_triangles_loss_full(loss_fn, sample_data):
    vertices, faces = sample_data

    simplified_data = {"sampled_vertices": vertices, "simplified_faces": faces}

    # Run the forward pass of the loss function
    loss = loss_fn(simplified_data)

    # Check that the loss is a scalar
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "Loss should be a scalar"

    # Expected behavior: the loss should be non-negative
    assert loss.item() >= 0, "Overlap loss should be non-negative"
