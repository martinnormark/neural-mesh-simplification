import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_add


class TriConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Update the input size of the MLP
        self.mlp = nn.Sequential(
            nn.Linear(
                in_channels + 9, out_channels
            ),  # 9 for relative position encoding
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, pos, edge_index):
        row, col = edge_index

        # Compute relative position encoding
        rel_pos = self.compute_relative_position_encoding(pos, row, col)

        # Compute feature differences
        x_diff = x[row] - x[col]

        # Concatenate relative position encoding and feature differences
        mlp_input = torch.cat([rel_pos, x_diff], dim=-1)

        # Apply MLP
        mlp_output = self.mlp(mlp_input)

        # Aggregate using sum
        out = scatter_add(mlp_output, col, dim=0, dim_size=x.size(0))

        return out

    def compute_relative_position_encoding(self, pos, row, col):
        # Compute edge vectors
        edge_vec = pos[row] - pos[col]

        # Compute t_max and t_min
        t_max, _ = scatter_max(edge_vec.abs(), col, dim=0, dim_size=pos.size(0))
        t_min, _ = scatter_max(-edge_vec.abs(), col, dim=0, dim_size=pos.size(0))
        t_min = -t_min  # Negate the result to get the minimum

        # Compute barycenter differences
        barycenter = pos.mean(dim=1, keepdim=True)
        barycenter_diff = barycenter[row] - barycenter[col]

        # Concatenate t_max, t_min, and barycenter differences
        rel_pos = torch.cat(
            [t_max[row] - t_max[col], t_min[row] - t_min[col], edge_vec], dim=-1
        )  # Use edge_vec instead of barycenter_diff

        return rel_pos
