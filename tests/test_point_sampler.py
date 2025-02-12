import dgl
import pytest
import torch
from dgl.nn.pytorch import GraphConv
from torch import nn

from neural_mesh_simplification.models.point_sampler import PointSamplerDGL


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


def test_point_sampler_initialization():
    sampler = PointSamplerDGL(in_channels=3, out_channels=64, num_layers=3)
    assert len(sampler.layers) == 3
    assert isinstance(sampler.layers[0], GraphConv)
    assert isinstance(sampler.output_layer, nn.Linear)


def test_point_sampler_forward(sample_graph):
    sampler = PointSamplerDGL(in_channels=3, out_channels=64, num_layers=3)
    probabilities = sampler(sample_graph)
    assert probabilities.shape == (4,)  # 4 input vertices
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_point_sampler_sampling(sample_graph):
    sampler = PointSamplerDGL(in_channels=3, out_channels=64, num_layers=3)
    probabilities = sampler(sample_graph)
    sampled_indices = sampler.sample(probabilities, num_samples=2)
    assert sampled_indices.shape == (2,)
    assert len(torch.unique(sampled_indices)) == 2  # All indices should be unique


def test_point_sampler_forward_and_sample(sample_graph):
    sampler = PointSamplerDGL(in_channels=3, out_channels=64, num_layers=3)
    probabilities = sampler.forward(sample_graph)
    sampled_indices = sampler.sample(probabilities, num_samples=2)
    assert sampled_indices.shape == (2,)
    assert probabilities.shape == (4,)
    assert len(torch.unique(sampled_indices)) == 2


def test_point_sampler_deterministic_behavior(sample_graph):
    sampler = PointSamplerDGL(in_channels=3, out_channels=64, num_layers=3)

    torch.manual_seed(42)
    probabilities1 = sampler.forward(sample_graph)
    indices1 = sampler.sample(probabilities1, num_samples=2)

    torch.manual_seed(42)
    probabilities2 = sampler.forward(sample_graph)
    indices2 = sampler.sample(probabilities2, num_samples=2)

    assert torch.equal(indices1, indices2)
