import torch.nn as nn
from torch_scatter import scatter_max


class DevConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__()
        self.W_theta = nn.Linear(in_channels, out_channels, bias=False)
        self.W_phi = nn.Linear(out_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]

        row, col = edge_index

        # Compute relative positions
        rel_pos = x[row] - x[col]  # [num_edges, in_channels]

        # Apply W_theta
        rel_pos_transformed = self.W_theta(rel_pos)  # [num_edges, out_channels]

        # Aggregate using max operation
        aggr_out = scatter_max(rel_pos_transformed, row, dim=0, dim_size=x.size(0))[0]

        # Apply W_phi
        out = self.W_phi(aggr_out)  # [num_nodes, out_channels]

        return out
