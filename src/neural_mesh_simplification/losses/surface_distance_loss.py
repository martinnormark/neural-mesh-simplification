import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProbabilisticSurfaceDistanceLoss(nn.Module):
    def __init__(self, num_samples: int = 100, epsilon: float = 1e-8):
        super().__init__()
        self.num_samples = num_samples
        self.epsilon = epsilon

    def forward(
            self,
            original_vertices: torch.Tensor,
            original_faces: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:

        logger.debug(f"Calculating SURFACE loss")
        logger.debug(
            f"devices (original_vertices, original_faces, simplified_vertices, simplified_faces, face_probabilities) = "
            f"({original_vertices}, {original_faces}, {simplified_vertices}, {simplified_faces}, {face_probabilities})"
        )

        # Early exit for empty meshes
        if original_vertices.shape[0] == 0 or simplified_vertices.shape[0] == 0:
            return torch.tensor(0.0, device=original_vertices.device)

        # Pad face probabilities once for both terms
        face_probabilities = torch.nn.functional.pad(
            face_probabilities,
            (0, max(0, simplified_faces.shape[0] - face_probabilities.shape[0]))
        )[:simplified_faces.shape[0]]

        # Compute forward and reverse terms
        forward_term = self.compute_forward_term(
            original_vertices, original_faces,
            simplified_vertices, simplified_faces,
            face_probabilities
        )

        reverse_term = self.compute_reverse_term(
            original_vertices,
            simplified_vertices,
            simplified_faces,
            face_probabilities
        )

        return forward_term + reverse_term

    def compute_forward_term(
            self,
            original_vertices: torch.Tensor,
            original_faces: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # Compute barycenters in batches to save memory
        batch_size = 1024
        num_faces = simplified_faces.shape[0]

        total_loss = torch.tensor(0.0, device=original_vertices.device)

        for i in range(0, num_faces, batch_size):
            batch_faces = simplified_faces[i:i + batch_size]
            batch_probs = face_probabilities[i:i + batch_size]

            # Compute barycenters for batch
            batch_barycenters = simplified_vertices[batch_faces].mean(dim=1)

            # Compute distances efficiently using cdist
            distances = torch.cdist(
                batch_barycenters,
                original_vertices[original_faces].mean(dim=1)
            )

            # Compute min distances and accumulate loss
            min_distances = distances.min(dim=1)[0]
            total_loss += (batch_probs * min_distances).sum()

        # Add probability penalty
        probability_penalty = 1e-4 * (1.0 - face_probabilities).sum()

        return total_loss + probability_penalty

    def compute_reverse_term(
            self,
            original_vertices: torch.Tensor,
            simplified_vertices: torch.Tensor,
            simplified_faces: torch.Tensor,
            face_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        # Sample points efficiently using vectorized operations
        num_faces = simplified_faces.shape[0]

        # Generate random values for barycentric coordinates
        r1 = torch.rand(num_faces, self.num_samples, 1, device=simplified_vertices.device)
        r2 = torch.rand(num_faces, self.num_samples, 1, device=simplified_vertices.device)

        # Compute barycentric coordinates
        sqrt_r1 = torch.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2

        # Get face vertices
        face_vertices = simplified_vertices[simplified_faces]

        # Compute sampled points using broadcasting
        sampled_points = (
                u * face_vertices[:, None, 0] +
                v * face_vertices[:, None, 1] +
                w * face_vertices[:, None, 2]
        )

        # Reshape sampled points
        sampled_points = sampled_points.reshape(-1, 3)

        # Compute distances efficiently using batched operations
        batch_size = 1024
        num_samples = sampled_points.shape[0]
        min_distances = torch.zeros(num_samples, device=simplified_vertices.device)

        for i in range(0, num_samples, batch_size):
            batch_points = sampled_points[i:i + batch_size]
            distances = torch.cdist(batch_points, original_vertices)
            min_distances[i:i + batch_size] = distances.min(dim=1)[0]

        # Scale distances
        max_dist = min_distances.max() + self.epsilon
        scaled_distances = (min_distances / max_dist) * 0.1

        # Compute final reverse term
        face_probs_expanded = face_probabilities.repeat_interleave(self.num_samples)
        reverse_term = (face_probs_expanded * scaled_distances).sum()

        return reverse_term
