import numpy as np
import trimesh
from data.dataset import preprocess_mesh


def test_preprocess_mesh():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    processed_mesh = preprocess_mesh(mesh)

    # Check that the mesh is centered
    assert np.allclose(
        processed_mesh.vertices.mean(axis=0), np.zeros(3)
    ), "Mesh is not centered"

    # Check that the mesh is scaled to unit cube
    max_dim = np.max(
        processed_mesh.vertices.max(axis=0) - processed_mesh.vertices.min(axis=0)
    )
    assert np.isclose(max_dim, 1.0), "Mesh is not scaled to unit cube"
