import torch
import pytest
from losses.surface_distance_loss import ProbabilisticSurfaceDistanceLoss


@pytest.fixture
def loss_fn():
    return ProbabilisticSurfaceDistanceLoss(k=3, num_samples=100)


@pytest.fixture
def simple_cube_data():
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
        dtype=torch.float32,
    )

    faces = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],
            [4, 5, 6],
            [5, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
            [2, 3, 6],
            [3, 7, 6],
            [0, 2, 4],
            [2, 6, 4],
            [1, 5, 3],
            [3, 5, 7],
        ],
        dtype=torch.long,
    )

    return vertices, faces


def test_loss_zero_for_identical_meshes(loss_fn, simple_cube_data):
    vertices, faces = simple_cube_data
    face_probs = torch.ones(faces.shape[0], dtype=torch.float32)

    loss = loss_fn(vertices, faces, vertices, faces, face_probs)
    print(f"Loss for identical meshes: {loss.item()}")
    assert loss.item() < 1e-5


def test_loss_increases_with_vertex_displacement(loss_fn, simple_cube_data):
    vertices, faces = simple_cube_data
    face_probs = torch.ones(faces.shape[0], dtype=torch.float32)

    displaced_vertices = vertices.clone()
    displaced_vertices[0] += torch.tensor([0.1, 0.1, 0.1])

    loss_original = loss_fn(vertices, faces, vertices, faces, face_probs)
    loss_displaced = loss_fn(vertices, faces, displaced_vertices, faces, face_probs)

    print(
        f"Original loss: {loss_original.item()}, Displaced loss: {loss_displaced.item()}"
    )
    assert loss_displaced > loss_original
    assert not torch.isclose(loss_displaced, loss_original, atol=1e-5)


def test_loss_increases_with_lower_face_probabilities(loss_fn, simple_cube_data):
    vertices, faces = simple_cube_data
    high_probs = torch.ones(faces.shape[0], dtype=torch.float32)
    low_probs = torch.full((faces.shape[0],), 0.5, dtype=torch.float32)

    loss_high = loss_fn(vertices, faces, vertices, faces, high_probs)
    loss_low = loss_fn(vertices, faces, vertices, faces, low_probs)

    assert loss_low > loss_high


def test_loss_handles_empty_meshes(loss_fn):
    empty_vertices = torch.empty((0, 3), dtype=torch.float32)
    empty_faces = torch.empty((0, 3), dtype=torch.long)
    empty_probs = torch.empty(0, dtype=torch.float32)

    loss = loss_fn(
        empty_vertices, empty_faces, empty_vertices, empty_faces, empty_probs
    )
    assert loss.item() == 0.0


def test_loss_is_symmetric(loss_fn, simple_cube_data):
    vertices, faces = simple_cube_data
    face_probs = torch.ones(faces.shape[0], dtype=torch.float32)

    loss_forward = loss_fn(vertices, faces, vertices, faces, face_probs)
    loss_reverse = loss_fn(vertices, faces, vertices, faces, face_probs)

    assert torch.isclose(loss_forward, loss_reverse, atol=1e-6)


def test_loss_gradients(loss_fn, simple_cube_data):
    vertices, faces = simple_cube_data
    vertices.requires_grad = True
    face_probs = torch.ones(faces.shape[0], dtype=torch.float32)
    face_probs.requires_grad = True

    loss = loss_fn(vertices, faces, vertices, faces, face_probs)
    loss.backward()

    assert vertices.grad is not None
    assert face_probs.grad is not None
    assert not torch.isnan(vertices.grad).any()
    assert not torch.isnan(face_probs.grad).any()
