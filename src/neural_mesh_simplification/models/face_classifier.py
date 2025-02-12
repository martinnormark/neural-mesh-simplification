import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class FaceClassifierDGL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, k):
        super(FaceClassifierDGL, self).__init__()
        self.k = k
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, g: dgl.DGLGraph, triangle_centers: torch.Tensor) -> torch.Tensor:
        # Create k-nn graph based on triangle centers
        knn_g = dgl.knn_graph(triangle_centers, k=self.k)

        h = g.ndata['x']
        for layer in self.layers:
            h = layer(knn_g, h)
            h = torch.relu(h)

        logits = self.final_layer(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs
