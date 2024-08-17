import torch
import torch.nn as nn
from .layers.devconv import DevConv


class PointSampler(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_layers=3):
        super(PointSampler, self).__init__()
        self.num_layers = num_layers

        # Stack of DevConv layers
        self.convs = nn.ModuleList()
        self.convs.append(DevConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.convs.append(DevConv(out_channels, out_channels))

        # Final output layer to produce a single score per vertex
        self.output_layer = nn.Linear(out_channels, 1)

        # Activation functions
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]

        # Apply DevConv layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)

        # Generate inclusion scores
        scores = self.output_layer(x).squeeze(-1)

        # Convert scores to probabilities
        probabilities = self.sigmoid(scores)

        return probabilities

    def sample(self, probabilities, num_samples):
        # Multinomial sampling based on probabilities
        sampled_indices = torch.multinomial(
            probabilities, num_samples, replacement=False
        )
        return sampled_indices

    def forward_and_sample(self, x, edge_index, num_samples):
        # Combine forward pass and sampling in one step
        probabilities = self.forward(x, edge_index)
        sampled_indices = self.sample(probabilities, num_samples)
        return sampled_indices, probabilities
