import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add


class TriConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 9, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.last_edge_index = None

    def forward(self, x, pos, edge_index):
        self.last_edge_index = edge_index
        row, col = edge_index

        rel_pos = self.compute_relative_position_encoding(pos, row, col)
        x_diff = x[row] - x[col]
        mlp_input = torch.cat([rel_pos, x_diff], dim=-1)

        mlp_output = self.mlp(mlp_input)
        out = scatter_add(mlp_output, col, dim=0, dim_size=x.size(0))

        return out

    def compute_relative_position_encoding(self, pos, row, col):
        edge_vec = pos[row] - pos[col]

        t_max, _ = scatter_max(edge_vec.abs(), col, dim=0, dim_size=pos.size(0))
        t_min, _ = scatter_max(-edge_vec.abs(), col, dim=0, dim_size=pos.size(0))
        t_min = -t_min

        barycenter = pos.mean(dim=1, keepdim=True)
        barycenter_diff = barycenter[row] - barycenter[col]

        t_max_diff = t_max[row] - t_max[col]
        t_min_diff = t_min[row] - t_min[col]
        barycenter_diff = (
            barycenter_diff.squeeze(-1).unsqueeze(-1).expand_as(t_max_diff)
        )

        rel_pos = torch.cat([t_max_diff, t_min_diff, barycenter_diff], dim=-1)

        return rel_pos
