import torch.nn as nn
from torch_scatter import scatter_max


class DevConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__()
        self.W_theta = nn.Linear(in_channels, out_channels)
        self.W_phi = nn.Linear(
            out_channels, out_channels
        )  # Fix: Apply after aggregation

    def forward(self, x, edge_index):
        row, col = edge_index  # Extract edge pairs
        x_i, x_j = x[row], x[col]  # Neighbor pairs

        rel_pos = x_i - x_j  # Compute relative position
        rel_pos_transformed = self.W_theta(rel_pos)  # Transform relative positions

        # Max aggregation over neighbors
        aggr_out = scatter_max(rel_pos_transformed, col, dim=0, dim_size=x.size(0))[0]

        # Apply transformation after aggregation (Aligns with equation)
        aggr_out = self.W_phi(aggr_out)

        return aggr_out
