import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv


class PointSamplerDGL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int):
        super(PointSamplerDGL, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_channels, out_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(out_channels, out_channels))
        self.output_layer = nn.Linear(out_channels, 1)

    def forward(self, g: DGLGraph) -> torch.Tensor:
        h = g.ndata['x']
        for layer in self.layers:
            h = layer(g, h)
            h = torch.relu(h)
        scores = self.output_layer(h).squeeze(-1)
        probabilities = torch.sigmoid(scores)
        return probabilities

    def sample(self, probabilities, num_samples):
        max_samples = probabilities.shape[0]
        if num_samples > max_samples:
            raise ValueError(
                f"num_samples ({num_samples}) cannot be larger than number of vertices ({max_samples})"
            )

        # Multinomial sampling based on probabilities
        sampled_indices = torch.multinomial(
            probabilities, num_samples, replacement=False
        )

        return sampled_indices
