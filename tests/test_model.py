import dgl
import pytest
import torch

from neural_mesh_simplification.models import NeuralMeshSimplification


@pytest.fixture
def sample_graph() -> dgl.DGLGraph:
    num_nodes = 4
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    pos = torch.randn(num_nodes, 3)
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    g.ndata['x'] = x
    g.ndata['pos'] = pos
    return g


def test_neural_mesh_simplification_forward(sample_graph: dgl.DGLGraph):
    # Set a fixed random seed for reproducibility
    torch.manual_seed(42)

    model = NeuralMeshSimplification(
        input_dim=3,
        hidden_dim=64,
        edge_hidden_dim=64,
        num_layers=3,
        k=3,  # Reduce k to avoid too many edges in the test
        edge_k=15,
        target_ratio=0.5  # Ensure we sample roughly half the vertices
    )

    # First test point sampling
    sampled_indices, sampled_probs = model.sample_points(sample_graph)
    assert sampled_indices.numel() > 0, "No points were sampled"
    assert sampled_indices.numel() <= sample_graph.num_nodes(), "Too many points sampled"

    sampled_sample_graph = dgl.node_subgraph(sample_graph, sampled_indices)
    assert sampled_sample_graph.num_nodes() > 0, "No nodes in sampled subgraph"
    assert sampled_sample_graph.num_edges() > 0, "No edges in sampled subgraph"

    # Now test the full forward pass
    simplified_g, simplified_faces, face_probs = model(sample_graph)

    # Add assertions to check the output structure and shapes
    assert isinstance(simplified_g, dgl.DGLGraph)
    assert isinstance(simplified_faces, torch.Tensor)
    assert isinstance(face_probs, torch.Tensor)

    assert simplified_g.num_nodes() <= sample_graph.num_nodes(), "Simplified graph has more nodes than original"
    assert simplified_g.num_edges() <= sample_graph.num_edges(), "Simplified graph has more edges than original"

    # Check that sampled_vertices correspond to a subset of original vertices
    original_vertices = sample_graph.ndata['x']
    sampled_vertices = simplified_g.ndata['x']

    # For each sampled vertex, check if it exists in original vertices
    for sv in sampled_vertices:
        # Check if this vertex exists in original vertices (within numerical precision)
        exists = torch.any(torch.all(torch.abs(original_vertices - sv) < 1e-6, dim=1))
        assert exists, "Sampled vertex not found in original vertices"

    # Check that simplified_faces only contain valid indices if not empty
    if simplified_faces.numel() > 0:
        max_index = simplified_g.ndata['x'].shape[0] - 1
        assert torch.all(simplified_faces >= 0)
        assert torch.all(simplified_faces <= max_index)

    # Check the relationship between face_probs and simplified_faces
    if face_probs.numel() > 0:
        assert simplified_faces.shape[0] <= face_probs.shape[0]


def test_generate_candidate_triangles():
    model = NeuralMeshSimplification(
        input_dim=3,
        hidden_dim=64,
        edge_hidden_dim=64,
        num_layers=3,
        k=5,
        edge_k=15,
        target_ratio=0.5
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    edge_probs = torch.tensor([0.9, 0.9, 0.8, 0.8, 0.7, 0.7])
    g = dgl.graph((edge_index[0], edge_index[1]))

    triangles, triangle_probs = model.generate_candidate_triangles(
        g, edge_probs
    )

    assert triangles.shape[1] == 3
    assert triangle_probs.shape[0] == triangles.shape[0]
    assert torch.all(triangles >= 0)
    assert torch.all(triangles < edge_index.max() + 1)
    assert torch.all(triangle_probs >= 0) and torch.all(triangle_probs <= 1)

    max_possible_triangles = edge_index.max().item() + 1  # num_nodes
    max_possible_triangles = (
            max_possible_triangles
            * (max_possible_triangles - 1)
            * (max_possible_triangles - 2)
            // 6
    )
    assert triangles.shape[0] <= max_possible_triangles
