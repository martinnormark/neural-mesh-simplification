import torch
import torch.nn as nn


class ProbabilisticSurfaceDistanceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        original_vertices: torch.Tensor,
        original_faces: torch.Tensor,
        simplified_vertices: torch.Tensor,
        simplified_faces: torch.Tensor,
        face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if original_vertices.shape[0] == 0 or simplified_vertices.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        # Step 1: Compute barycenters of both original and simplified meshes
        original_barycenters = self.compute_barycenters(
            original_vertices, original_faces
        )
        simplified_barycenters = self.compute_barycenters(
            simplified_vertices, simplified_faces
        )

        # Step 2: Calculate the squared distances between each simplified barycenter and all original barycenters
        distances = self.compute_squared_distances(
            simplified_barycenters, original_barycenters
        )

        # Step 3: Find the minimum distance for each simplified barycenter
        min_distances, _ = distances.min(dim=1)

        # Step 4: Weight by face probabilities and sum
        weighted_distances = face_probabilities * min_distances
        total_loss = weighted_distances.sum()

        return total_loss

    @staticmethod
    def compute_barycenters(
        vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        return vertices[faces].mean(dim=1)

    @staticmethod
    def compute_squared_distances(
        barycenters1: torch.Tensor, barycenters2: torch.Tensor
    ) -> torch.Tensor:
        # barycenters1: (num_faces1, 3)
        # barycenters2: (num_faces2, 3)

        num_faces1 = barycenters1.size(0)
        num_faces2 = barycenters2.size(0)

        # Expand dimensions to compute pairwise differences
        barycenters1_exp = barycenters1.unsqueeze(1).expand(num_faces1, num_faces2, 3)
        barycenters2_exp = barycenters2.unsqueeze(0).expand(num_faces1, num_faces2, 3)

        # Compute squared Euclidean distances
        distances = torch.sum((barycenters1_exp - barycenters2_exp) ** 2, dim=2)

        return distances
