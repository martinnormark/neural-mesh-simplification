import trimesh
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from data.dataset import (
    MeshSimplificationDataset,
    preprocess_mesh,
    mesh_to_tensor,
    load_mesh,
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


def test_mesh_to_tensor(sample_mesh: trimesh.Trimesh):
    data = mesh_to_tensor(sample_mesh)
    assert isinstance(data, Data)
    assert data.num_nodes == len(sample_mesh.vertices)
    assert data.face.shape[1] == len(sample_mesh.faces)
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.max() < data.num_nodes


def test_graph_structure_in_data(sample_mesh):
    data = mesh_to_tensor(sample_mesh)

    # Check number of nodes
    assert data.num_nodes == len(sample_mesh.vertices)

    # Check edge_index
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.max() < data.num_nodes

    # Reconstruct graph from edge_index
    G = nx.Graph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    # Check reconstructed graph properties
    assert len(G.nodes) == len(sample_mesh.vertices)
    assert len(G.edges) == (3 * len(sample_mesh.faces) - len(sample_mesh.edges_unique))

    # Check connectivity
    assert nx.is_connected(G)

    # Check degree distribution
    degrees = [d for n, d in G.degree()]
    assert min(degrees) >= 3  # Each vertex should be connected to at least 3 others

    # Check if the graph is manifold-like (each edge should be shared by at most two faces)
    edge_face_count = {}
    for face in sample_mesh.faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
    assert all(count <= 2 for count in edge_face_count.values())


def test_dataset(tmp_path):
    # Create a few temporary mesh files
    for i in range(3):
        mesh = trimesh.creation.box()
        file_path = tmp_path / f"test_mesh_{i}.obj"
        mesh.export(file_path)

    dataset = MeshSimplificationDataset(str(tmp_path))
    assert len(dataset) == 3

    sample = dataset[0]
    assert isinstance(sample, Data)
    assert sample.num_nodes > 0
    assert sample.face.shape[1] > 0
