import dgl
import numpy as np
import torch
import trimesh
from numpy.testing import assert_array_equal

from neural_mesh_simplification.data.dataset import (
    preprocess_mesh,
    load_mesh, mesh_to_dgl,
    dgl_to_trimesh, collate
)


def test_load_mesh(tmp_path):
    # Create a temporary mesh file
    mesh = trimesh.creation.box()
    file_path = tmp_path / "test_mesh.obj"
    mesh.export(file_path)

    loaded_mesh = load_mesh(str(file_path))
    assert isinstance(loaded_mesh, trimesh.Trimesh)
    assert np.allclose(loaded_mesh.vertices, mesh.vertices)
    assert np.array_equal(loaded_mesh.faces, mesh.faces)


def test_preprocess_mesh_centered(sample_mesh):
    processed_mesh = preprocess_mesh(sample_mesh)
    # Check that the mesh is centered
    assert np.allclose(
        processed_mesh.vertices.mean(axis=0), np.zeros(3)
    ), "Mesh is not centered"


def test_preprocess_mesh_scaled(sample_mesh):
    processed_mesh = preprocess_mesh(sample_mesh)

    max_dim = np.max(
        processed_mesh.vertices.max(axis=0) - processed_mesh.vertices.min(axis=0)
    )
    assert np.isclose(max_dim, 1.0), "Mesh is not scaled to unit cube"


def test_face_serde(sample_mesh):
    orig_faces = torch.tensor(sample_mesh.faces, dtype=torch.int64)
    g, faces = mesh_to_dgl(sample_mesh)

    assert torch.equal(orig_faces, faces)

    r_mesh = dgl_to_trimesh(g, faces)

    assert_array_equal(r_mesh.vertices, sample_mesh.vertices)
    assert_array_equal(r_mesh.faces, sample_mesh.faces)


def test_padding():
    g1 = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
    g2 = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=3)
    f1 = torch.tensor([[0, 1, 2], [1, 2, 0]])
    f2 = torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    batch = [(g1, f1), (g2, f2)]

    # Pad
    _, padded_faces = collate(batch)

    # Unpad
    unpadded_faces = [f[~(f == -1).all(dim=1)] for f in padded_faces]

    # Assert idempotency
    for original, unpadded in zip([f1, f2], unpadded_faces):
        assert torch.all(original == unpadded)
        assert original.shape == unpadded.shape
