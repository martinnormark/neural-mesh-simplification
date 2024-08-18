import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
from .layers import TriConv


class FaceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, k=20):
        super(FaceClassifier, self).__init__()
        self.k = k
        self.num_layers = num_layers

        self.triconv_layers = nn.ModuleList(
            [
                TriConv(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )

        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, pos, batch=None):
        # Compute barycenters if pos is 3D
        if pos.dim() == 3:
            pos = pos.mean(dim=1)

        # Construct k-nn graph based on barycenter distances
        edge_index = self.custom_knn_graph(pos, self.k, batch)

        # Apply TriConv layers
        for i in range(self.num_layers):
            x = self.triconv_layers[i](x, pos, edge_index)
            x = torch.relu(x)

        # Final classification
        x = self.final_layer(x).squeeze(-1)

        # Apply softmax per batch
        if batch is None:
            probabilities = torch.softmax(x, dim=0)
        else:
            probabilities = scatter_softmax(x, batch, dim=0)

        return probabilities

    def custom_knn_graph(self, x, k, batch=None):
        batch_size = 1 if batch is None else batch.max().item() + 1
        edge_index = []

        for b in range(batch_size):
            if batch is None:
                x_batch = x
            else:
                mask = batch == b
                x_batch = x[mask]

            distances = torch.cdist(x_batch, x_batch)
            distances.fill_diagonal_(float("inf"))
            _, indices = distances.topk(k, largest=False)

            source = (
                torch.arange(x_batch.size(0), device=x.device).view(-1, 1).expand(-1, k)
            )
            edge_index.append(torch.stack([source.reshape(-1), indices.reshape(-1)]))

        edge_index = torch.cat(edge_index, dim=1)

        # Make the graph symmetric
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index = torch.unique(edge_index, dim=1)

        return edge_index
