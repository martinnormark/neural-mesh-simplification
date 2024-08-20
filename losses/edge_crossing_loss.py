import torch.nn as nn


class EdgeCrossingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, original_data, simplified_data):
        pass
