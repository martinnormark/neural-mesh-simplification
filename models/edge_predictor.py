import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax
from torch_sparse import SparseTensor
from .layers.devconv import DevConv


class EdgePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, k=15):
        super(EdgePredictor, self).__init__()
        self.k = k
        self.devconv = DevConv(in_channels, hidden_channels)

        # Self-attention components
        self.W_q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_k = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x, edge_index):
        # Handle the case when edge_index is empty
        if edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=x.device), torch.empty(
                0, device=x.device
            )

        # Step 1: Extend original mesh connectivity
        knn_edges = knn_graph(x, k=self.k, flow="target_to_source")
        extended_edges = torch.cat([edge_index, knn_edges], dim=1)

        # Step 2: Apply DevConv
        features = self.devconv(x, extended_edges)

        # Step 3: Apply sparse self-attention
        attention_scores = self.compute_attention_scores(features, edge_index)

        # Step 4: Compute simplified adjacency matrix
        simplified_adj_indices, simplified_adj_values = (
            self.compute_simplified_adjacency(attention_scores, edge_index)
        )

        return simplified_adj_indices, simplified_adj_values

    def compute_attention_scores(self, features, edges):
        row, col = edges
        q = self.W_q(features)
        k = self.W_k(features)

        # Compute (W_q f_j)^T (W_k f_i)
        attention = (q[row] * k[col]).sum(dim=-1)

        # Apply softmax for each source node
        attention_scores = scatter_softmax(attention, row, dim=0)

        return attention_scores

    def compute_simplified_adjacency(self, attention_scores, edge_index):
        if edge_index.numel() == 0:
            return torch.empty(
                (2, 0), dtype=torch.long, device=edge_index.device
            ), torch.empty(0, device=edge_index.device)

        num_nodes = attention_scores.size(0)

        # Remap node indices to ensure they are within the correct range
        row, col = edge_index
        row_offset = row.min()
        col_offset = col.min()
        row = row - row_offset
        col = col - col_offset

        # Create sparse attention matrix
        S = SparseTensor(
            row=row,
            col=col,
            value=attention_scores,
            sparse_sizes=(num_nodes - row_offset, num_nodes - col_offset),
        )

        # Create original adjacency matrix
        A = SparseTensor(
            row=row,
            col=col,
            value=torch.ones(edge_index.size(1), device=edge_index.device),
            sparse_sizes=(num_nodes - row_offset, num_nodes - col_offset),
        )

        # Compute A_s = S * A * S^T
        A_s = S @ A @ S.t()

        # Convert to COO format
        row, col, value = A_s.coo()
        indices = torch.stack([row + row_offset, col + col_offset], dim=0)

        return indices, value
