import pytest
import torch
from losses.triangle_collision_loss import TriangleCollisionLoss


@pytest.fixture
def collision_loss():
    return TriangleCollisionLoss(k=1, collision_threshold=1e-6)


@pytest.fixture
def test_meshes():
    return {
        "non_penetrating": {
            "vertices": torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],  # Triangle 1
                    [1, 1, 0],  # Additional vertex for Triangle 2
                ],
                dtype=torch.float32,
            ),
            "faces": torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.long),
        },
        "edge_penetrating": {
            "vertices": torch.tensor(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0.5, 1, 0],  # Triangle 1
                    [0.25, 0.25, -0.5],
                    [0.75, 0.25, 0.5],
                    [0.5, 0.75, 0],  # Triangle 2
                ],
                dtype=torch.float32,
            ),
            "faces": torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        },
    }


def test_collision_detection(collision_loss, test_meshes):
    for name, data in test_meshes.items():
        face_probabilities = torch.ones(data["faces"].shape[0], dtype=torch.float32)
        loss = collision_loss(data["vertices"], data["faces"], face_probabilities)

        print(f"{name} - Loss: {loss.item()}")

        if name == "non_penetrating":
            assert (
                loss.item() == 0
            ), f"Non-zero loss computed for non-penetrating case: {name}"
        else:
            assert loss.item() > 0, f"Zero loss computed for penetrating case: {name}"


def test_collision_loss_values(collision_loss, test_meshes):
    data = test_meshes["edge_penetrating"]
    face_probabilities = torch.ones(data["faces"].shape[0], dtype=torch.float32)
    loss = collision_loss(data["vertices"], data["faces"], face_probabilities)

    assert (
        loss.item() > 0
    ), f"Expected positive loss for edge penetrating case, got {loss.item()}"
    print(f"Edge penetrating loss: {loss.item()}")


def test_collision_loss_with_probabilities(collision_loss, test_meshes):
    data = test_meshes["edge_penetrating"]
    face_probabilities = torch.tensor([0.5, 0.7], dtype=torch.float32)
    loss = collision_loss(data["vertices"], data["faces"], face_probabilities)

    assert (
        loss.item() > 0
    ), f"Expected positive loss for edge penetrating case with probabilities, got {loss.item()}"
    print(f"Edge penetrating loss with probabilities: {loss.item()}")


def test_empty_mesh(collision_loss):
    empty_vertices = torch.empty((0, 3), dtype=torch.float32)
    empty_faces = torch.empty((0, 3), dtype=torch.long)
    empty_probabilities = torch.empty(0, dtype=torch.float32)

    loss = collision_loss(empty_vertices, empty_faces, empty_probabilities)
    assert loss.item() == 0, f"Expected zero loss for empty mesh, got {loss.item()}"


@pytest.fixture
def complex_mesh():
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
        ],
        dtype=torch.long,
    )
    return {"vertices": vertices, "faces": faces}


def test_complex_mesh(collision_loss, complex_mesh):
    face_probabilities = torch.ones(complex_mesh["faces"].shape[0], dtype=torch.float32)
    loss = collision_loss(
        complex_mesh["vertices"], complex_mesh["faces"], face_probabilities
    )
    print(f"Complex mesh - Loss: {loss.item()}")
    assert loss.item() >= 0, "Negative loss computed for complex mesh"


def test_collision_detection_edge_cases(collision_loss):
    vertices = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],  # Triangle 1
            [0, 0, 1e-7],
            [1, 0, 1e-7],
            [0, 1, 1e-7],  # Triangle 2 (very close but not intersecting)
            [0.5, 0.5, -0.5],
            [1.5, 0.5, -0.5],
            [0.5, 1.5, -0.5],  # Triangle 3 (intersecting)
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.long)
    face_probabilities = torch.ones(faces.shape[0], dtype=torch.float32)

    # Calculate and print face normals
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = torch.linalg.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-6)
    print("Face normals:")
    print(face_normals)

    # Print triangle information
    for i, face in enumerate(faces):
        print(f"Triangle {i}:")
        print(f"  Vertices: {vertices[face].tolist()}")
        print(f"  Normal: {face_normals[i].tolist()}")
        print(f"  Centroid: {vertices[face].mean(dim=0).tolist()}")

    loss = collision_loss(vertices, faces, face_probabilities)
    print(f"Edge case loss: {loss.item()}")

    assert loss.item() > 0, "Should detect the intersecting triangle"
    assert (
        loss.item() < 3
    ), "Should not detect collision for very close but non-intersecting triangles"
    assert (
        loss.item() == 2
    ), "Should detect exactly two collisions (Triangle 3 intersects both Triangle 1 and 2)"
