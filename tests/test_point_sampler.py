import pytest
import torch
from torch import nn
from models.layers.devconv import DevConv
from models.point_sampler import PointSampler


@pytest.fixture
def sample_graph_data():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    return x, edge_index


def test_point_sampler_initialization():
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)
    assert len(sampler.convs) == 3
    assert isinstance(sampler.convs[0], DevConv)
    assert isinstance(sampler.output_layer, nn.Linear)


def test_point_sampler_forward(sample_graph_data):
    x, edge_index = sample_graph_data
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)
    probabilities = sampler(x, edge_index)
    assert probabilities.shape == (4,)  # 4 input vertices
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_point_sampler_sampling(sample_graph_data):
    x, edge_index = sample_graph_data
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)
    probabilities = sampler(x, edge_index)
    sampled_indices = sampler.sample(probabilities, num_samples=2)
    assert sampled_indices.shape == (2,)
    assert len(torch.unique(sampled_indices)) == 2  # All indices should be unique


def test_point_sampler_forward_and_sample(sample_graph_data):
    x, edge_index = sample_graph_data
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)
    sampled_indices, probabilities = sampler.forward_and_sample(
        x, edge_index, num_samples=2
    )
    assert sampled_indices.shape == (2,)
    assert probabilities.shape == (4,)
    assert len(torch.unique(sampled_indices)) == 2


def test_point_sampler_deterministic_behavior(sample_graph_data):
    x, edge_index = sample_graph_data
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)

    torch.manual_seed(42)
    indices1, _ = sampler.forward_and_sample(x, edge_index, num_samples=2)

    torch.manual_seed(42)
    indices2, _ = sampler.forward_and_sample(x, edge_index, num_samples=2)

    assert torch.equal(indices1, indices2)


def test_point_sampler_different_input_sizes():
    sampler = PointSampler(in_channels=3, out_channels=64, num_layers=3)

    x1 = torch.rand(10, 3)
    edge_index1 = torch.randint(0, 10, (2, 20))
    prob1 = sampler(x1, edge_index1)
    assert prob1.shape == (10,)

    x2 = torch.rand(20, 3)
    edge_index2 = torch.randint(0, 20, (2, 40))
    prob2 = sampler(x2, edge_index2)
    assert prob2.shape == (20,)
