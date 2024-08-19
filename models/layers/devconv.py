import torch.nn as nn
from torch_scatter import scatter_max


class DevConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DevConv, self).__init__()
        self.W_theta = nn.Linear(in_channels, out_channels, bias=False)
        self.W_phi = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        rel_pos = x_i - x_j
        rel_pos_transformed = self.W_theta(rel_pos)  # [num_edges, out_channels]

        x_transformed = self.W_phi(x)  # [num_nodes, out_channels]

        # Aggregate using max pooling
        aggr_out = scatter_max(rel_pos_transformed, col, dim=0, dim_size=x.size(0))[0]

        return x_transformed + aggr_out
