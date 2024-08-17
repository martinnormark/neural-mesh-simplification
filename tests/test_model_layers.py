import sys
import pytest
import torch
from torch import nn
from models.layers import DevConv, TriConv


def create_graph_data():
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Node 0
            [1.0, 0.0, 0.0],  # Node 1
            [0.0, 1.0, 0.0],  # Node 2
            [1.0, 1.0, 0.0],  # Node 3
        ],
        dtype=torch.float,
    )

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],  # Source nodes
            [1, 2, 0, 3, 0, 3, 1, 2],  # Target nodes
        ],
        dtype=torch.long,
    )

    return x, edge_index


@pytest.fixture
def graph_data():
    return create_graph_data()


def test_devconv(graph_data):
    x, edge_index = graph_data

    devconv = DevConv(in_channels=3, out_channels=4)
    output = devconv(x, edge_index)

    assert output.shape == (4, 4)  # 4 nodes, 4 output channels

    if "-s" in sys.argv:
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)
        print("Output:\n", output)

        analyze_feature_differences(x, edge_index)


def analyze_feature_differences(x, edge_index):
    devconv = DevConv(in_channels=3, out_channels=3)
    output = devconv(x, edge_index)

    for i in range(x.shape[0]):
        neighbors = edge_index[1][edge_index[0] == i]
        print(f"Node {i}:")
        print(f"  Input feature: {x[i]}")
        print(f"  Output feature: {output[i]}")
        print("  Neighbor differences:")
        for j in neighbors:
            print(f"    Node {j}: {x[i] - x[j]}")
        print()


@pytest.fixture
def triconv_layer():
    return TriConv(in_channels=16, out_channels=32)


def test_triconv_initialization(triconv_layer):
    assert triconv_layer.in_channels == 16
    assert triconv_layer.out_channels == 32
    assert isinstance(triconv_layer.mlp, nn.Sequential)


def test_triconv_forward(triconv_layer):
    num_nodes = 10
    x = torch.randn(num_nodes, 16)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    out = triconv_layer(x, pos, edge_index)
    assert out.shape == (num_nodes, 32)


def test_relative_position_encoding(triconv_layer):
    num_nodes = 5
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    rel_pos = triconv_layer.compute_relative_position_encoding(
        pos, edge_index[0], edge_index[1]
    )
    assert rel_pos.shape == (4, 9)  # 4 edges, 9-dimensional encoding


def test_triconv_gradient(triconv_layer):
    num_nodes = 10
    x = torch.randn(num_nodes, 16, requires_grad=True)
    pos = torch.randn(num_nodes, 3, requires_grad=True)
    edge_index = torch.randint(0, num_nodes, (2, 20))

    out = triconv_layer(x, pos, edge_index)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert pos.grad is not None
    assert all(p.grad is not None for p in triconv_layer.parameters())
