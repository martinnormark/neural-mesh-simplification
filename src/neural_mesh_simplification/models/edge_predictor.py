import warnings

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


class EdgePredictorDGL(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, k: int):
        super(EdgePredictorDGL, self).__init__()
        self.k = k
        self.conv = GraphConv(in_channels, hidden_channels)
        self.W_q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_k = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, g: DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges and their probabilities.

        Args:
            g (dgl.DGLGraph): Input graph with node features 'x'.

        Returns:
            tuple: (edge_index_pred, edge_probs)
                - edge_index_pred (torch.Tensor): Predicted edges [2, num_edges].
                - edge_probs (torch.Tensor): Probabilities for each predicted edge [num_edges].
        """
        # Step 1: Apply graph convolution to compute node embeddings
        h = g.ndata['x']
        h = self.conv(g, h)

        # Step 2: Compute query (q) and key (k) vectors for attention
        q = self.W_q(h)
        k = self.W_k(h)

        # Step 3: Compute attention scores for all edges in the graph
        g.ndata['q'] = q
        g.ndata['k'] = k
        g.apply_edges(lambda edges: {'score': (edges.src['q'] * edges.dst['k']).sum(dim=-1)})

        # Step 4: Normalize scores using softmax to get edge probabilities
        g.edata['prob'] = dgl.nn.functional.edge_softmax(g, g.edata['score'])

        # Step 5: Extract predicted edges and their probabilities
        edge_index_pred = torch.stack(g.edges(), dim=0)  # Shape: [2, num_edges]
        edge_probs = g.edata['prob']  # Shape: [num_edges]

        return edge_index_pred, edge_probs
