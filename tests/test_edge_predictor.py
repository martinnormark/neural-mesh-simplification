import torch
import pytest
import torch.nn as nn
from torch_geometric.data import Data
from models.edge_predictor import EdgePredictor
from models.layers.devconv import DevConv


@pytest.fixture
def sample_mesh_data():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    return Data(x=x, edge_index=edge_index)


def test_edge_predictor_initialization():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=15)
    assert isinstance(edge_predictor.devconv, DevConv)
    assert isinstance(edge_predictor.W_q, nn.Linear)
    assert isinstance(edge_predictor.W_k, nn.Linear)
    assert edge_predictor.k == 15


def test_edge_predictor_forward(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    assert isinstance(simplified_adj_indices, torch.Tensor)
    assert isinstance(simplified_adj_values, torch.Tensor)
    assert simplified_adj_indices.shape[0] == 2  # 2 rows for source and target indices
    assert (
        simplified_adj_values.shape[0] == simplified_adj_indices.shape[1]
    )  # Same number of values as edges


def test_edge_predictor_output_range(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    _, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    assert (simplified_adj_values >= 0).all()  # Values should be non-negative


def test_edge_predictor_symmetry(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, simplified_adj_values = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    # Create a sparse tensor from the output
    n = sample_mesh_data.x.shape[0]
    adj_matrix = torch.sparse_coo_tensor(
        simplified_adj_indices, simplified_adj_values, (n, n)
    )
    dense_adj = adj_matrix.to_dense()

    assert torch.allclose(dense_adj, dense_adj.t(), atol=1e-6)


def test_edge_predictor_connectivity(sample_mesh_data):
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=2)
    simplified_adj_indices, _ = edge_predictor(
        sample_mesh_data.x, sample_mesh_data.edge_index
    )

    # Check if all nodes are connected
    unique_nodes = torch.unique(simplified_adj_indices)
    assert len(unique_nodes) == sample_mesh_data.x.shape[0]


def test_edge_predictor_different_input_sizes():
    edge_predictor = EdgePredictor(in_channels=3, hidden_channels=64, k=5)

    # Test with a larger graph
    x = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 30))
    simplified_adj_indices, simplified_adj_values = edge_predictor(x, edge_index)

    assert simplified_adj_indices.shape[0] == 2
    assert simplified_adj_values.shape[0] == simplified_adj_indices.shape[1]
    assert torch.max(simplified_adj_indices) < 10
