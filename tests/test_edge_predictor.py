import dgl
import pytest
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

from neural_mesh_simplification.models.edge_predictor import EdgePredictorDGL


@pytest.fixture
def sample_graph() -> dgl.DGLGraph:
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
    g.ndata['x'] = x
    return g


def test_edge_predictor_initialization():
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=15)
    assert isinstance(edge_predictor.conv, GraphConv)
    assert isinstance(edge_predictor.W_q, nn.Linear)
    assert isinstance(edge_predictor.W_k, nn.Linear)
    assert edge_predictor.k == 15


def test_edge_predictor_forward(sample_graph: dgl.DGLGraph):
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_graph
    )

    assert isinstance(simplified_adj_indices, torch.Tensor)
    assert isinstance(simplified_adj_values, torch.Tensor)
    assert simplified_adj_indices.shape[0] == 2  # 2 rows for source and target indices
    assert (
        simplified_adj_values.shape[0] == simplified_adj_indices.shape[1]
    )  # Same number of values as edges


def test_edge_predictor_output_range(sample_graph: dgl.DGLGraph):
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=2)
    _, simplified_adj_values = edge_predictor(
        sample_graph
    )

    assert (simplified_adj_values >= 0).all()  # Values should be non-negative


def test_edge_predictor_symmetry(sample_graph: dgl.DGLGraph):
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_graph
    )

    # Create a sparse tensor from the output
    n = sample_graph.num_nodes()
    adj_matrix = torch.sparse_coo_tensor(
        simplified_adj_indices, simplified_adj_values, (n, n)
    )
    dense_adj = adj_matrix.to_dense()

    assert torch.allclose(dense_adj, dense_adj.t(), atol=1e-6)


def test_edge_predictor_connectivity(sample_graph):
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, _ = edge_predictor(
        sample_graph
    )

    # Check if all nodes are connected
    unique_nodes = torch.unique(simplified_adj_indices)
    assert len(unique_nodes) == sample_graph.num_nodes()


def test_feature_transformation():
    edge_predictor = EdgePredictorDGL(in_channels=3, hidden_channels=64, k=2)
    x = torch.rand(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    # Get intermediate features
    g_knn = dgl.knn_graph(x, k=2)
    features = edge_predictor.conv(g_knn, x)

    # Check feature dimensions
    assert features.shape == (5, 64)  # [num_nodes, hidden_channels]

    # Check transformed features through attention layers
    q = edge_predictor.W_q(features)
    k = edge_predictor.W_k(features)
    assert q.shape == (5, 64)
    assert k.shape == (5, 64)
