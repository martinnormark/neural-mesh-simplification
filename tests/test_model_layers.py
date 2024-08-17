import sys
import torch
import pytest
from models.layers.devconv import DevConv


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
