import torch.nn as nn
from losses import (
    ProbabilisticChamferDistanceLoss,
    ProbabilisticSurfaceDistanceLoss,
    TriangleCollisionLoss,
    EdgeCrossingLoss,
    OverlapLoss,
)


class CombinedMeshSimplificationLoss(nn.Module):
    def __init__(
        self, lambda_c: float = 1.0, lambda_e: float = 1.0, lambda_o: float = 1.0
    ):
        super().__init__()
        self.prob_chamfer_loss = ProbabilisticChamferDistanceLoss()
        self.prob_surface_loss = ProbabilisticSurfaceDistanceLoss()
        self.collision_loss = TriangleCollisionLoss()
        self.edge_crossing_loss = EdgeCrossingLoss()
        self.overlap_loss = OverlapLoss()
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_o = lambda_o

    def forward(self, original_data, simplified_data):
        chamfer_loss = self.prob_chamfer_loss(original_data, simplified_data)
        surface_loss = self.prob_surface_loss(
            original_data["pos"],
            original_data["face"],
            simplified_data["sampled_vertices"],
            simplified_data["simplified_faces"],
            simplified_data["face_probs"],
        )
        collision_loss = self.collision_loss(simplified_data)
        edge_crossing_loss = self.edge_crossing_loss(simplified_data)
        overlap_loss = self.overlap_loss(simplified_data)

        total_loss = (
            chamfer_loss
            + surface_loss
            + self.lambda_c * collision_loss
            + self.lambda_e * edge_crossing_loss
            + self.lambda_o * overlap_loss
        )

        return total_loss
