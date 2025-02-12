import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OverlappingTrianglesLoss(nn.Module):
    def __init__(self, num_samples: int = 10, k: int = 5):
        """
        Initializes the OverlappingTrianglesLoss.

        Args:
            num_samples (int): The number of points to sample from each triangle.
            k (int): The number of nearest triangles to consider for overlap checking.
        """
        super().__init__()
        self.num_samples = num_samples  # Number of points to sample from each triangle
        self.k = k  # Number of nearest triangles to consider

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor):

        logger.debug(f"Calculating SURFACE loss")
        logger.debug(f"devices (vertices, faces) = ({vertices}, {faces})")

        # If no faces, return zero loss
        if faces.shape[0] == 0:
            return torch.tensor(0.0, device=vertices.device)

        # 1. Sample points from each triangle
        sampled_points, point_face_indices = self.sample_points_from_triangles(vertices, faces)

        # 2. Find k-nearest triangles for each point
        nearest_triangles = self.find_nearest_triangles(sampled_points, vertices, faces)

        # 3. Detect overlaps and calculate the loss
        overlap_penalty = self.calculate_overlap_loss(
            sampled_points, vertices, faces, nearest_triangles, point_face_indices
        )

        return overlap_penalty

    def sample_points_from_triangles(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Samples points from each triangle in the mesh.

        Args:
            vertices (torch.Tensor): The vertex positions (V x 3).
            faces (torch.Tensor): The indices of the vertices that make up each triangle (F x 3).

        Returns:
            torch.Tensor: Sampled points (F * num_samples x 3).
            torch.Tensor: index mapping from points to their original faces.
        """
        # Get vertices for all faces at once
        v0, v1, v2 = vertices[faces].unbind(1)

        # Generate random barycentric coordinates
        rand_shape = (faces.shape[0], self.num_samples, 1)
        u = torch.rand(rand_shape, device=vertices.device)
        v = torch.rand(rand_shape, device=vertices.device)

        # Adjust coordinates that sum > 1
        mask = (u + v) > 1
        u = torch.where(mask, 1 - u, u)
        v = torch.where(mask, 1 - v, v)
        w = 1 - u - v

        # Calculate the coordinates of the sampled points
        points = (
            v0.unsqueeze(1) * w +
            v1.unsqueeze(1) * u +
            v2.unsqueeze(1) * v
        )

        # Create index mapping from points to their original faces
        point_face_indices = torch.arange(faces.shape[0], device=vertices.device)
        point_face_indices = point_face_indices.repeat_interleave(self.num_samples)

        # Reshape to a (F * num_samples x 3) tensor
        points = points.reshape(-1, 3)

        return points, point_face_indices

    def find_nearest_triangles(
        self,
        sampled_points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Finds the k-nearest triangles for each sampled point.

        Args:
            sampled_points (torch.Tensor): Sampled points from triangles (N x 3).
            vertices (torch.Tensor): The vertex positions (V x 3).
            faces (torch.Tensor): The indices of the vertices that make up each triangle (F x 3).

        Returns:
            torch.Tensor: Indices of the k-nearest triangles for each sampled point (N x k).
        """
        # Compute triangle centroids
        centroids = vertices[faces].mean(dim=1)

        # Adjust k to be no larger than the number of triangles
        k = min(self.k, faces.shape[0])
        if k == 0:
            # Return empty tensor if no triangles
            return torch.empty(
                (sampled_points.shape[0], 0),
                dtype=torch.long,
                device=sampled_points.device,
            )

        # Use knn to find nearest triangles for each sampled point
        distances = torch.cdist(sampled_points, centroids)
        _, indices = distances.topk(k, dim=1, largest=False)

        return indices

    def calculate_overlap_loss(
        self,
        sampled_points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        nearest_triangles: torch.Tensor,
        point_face_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the overlap loss by checking if sampled points belong to multiple triangles.

        Args:
            sampled_points (torch.Tensor): Sampled points from triangles (N x 3).
            vertices (torch.Tensor): The vertex positions (V x 3).
            faces (torch.Tensor): The indices of the vertices that make up each triangle (F x 3).
            nearest_triangles (torch.Tensor): Indices of the k-nearest triangles for each sampled point (N x k).
            point_face_indices (torch.Tensor): Index mapping from points to their original faces.

        Returns:
            torch.Tensor: The overlap penalty loss.
        """

        # Reshape for broadcasting
        points_expanded = sampled_points.unsqueeze(1)  # [N, 1, 3]
        nearest_faces = faces[nearest_triangles]  # [N, K, 3]

        # Get vertices for all nearest triangles
        v0 = vertices[nearest_faces[..., 0]]  # [N, K, 3]
        v1 = vertices[nearest_faces[..., 1]]
        v2 = vertices[nearest_faces[..., 2]]

        # Calculate edges
        edge1 = v1 - v0  # [N, K, 3]
        edge2 = v2 - v0

        # Calculate normals
        normals = torch.linalg.cross(edge1, edge2)  # [N, K, 3]
        normal_lengths = torch.norm(normals, dim=2, keepdim=True)
        normals = normals / (normal_lengths + 1e-8)

        del edge1, edge2, normal_lengths

        # Calculate barycentric coordinates for all points at once
        p_v0 = points_expanded - v0  # [N, K, 3]

        # Compute dot products for barycentric coordinates
        dot00 = torch.sum(normals * normals, dim=2)  # [N, K]
        dot01 = torch.sum(normals * (v1 - v0), dim=2)
        dot02 = torch.sum(normals * (v2 - v0), dim=2)
        dot0p = torch.sum(normals * p_v0, dim=2)

        del p_v0, normals

        # Calculate barycentric coordinates
        denom = (dot00 * dot00 - dot01 * dot01)
        u = (dot00 * dot0p - dot01 * dot02) / (denom + 1e-8)
        v = (dot00 * dot02 - dot01 * dot0p) / (denom + 1e-8)

        del dot00, dot01, dot02, dot0p, denom

        # Check if points are inside triangles
        inside_mask = (u >= 0) & (v >= 0) & (u + v <= 1)

        # Don't count overlap with source triangle
        source_mask = nearest_triangles == point_face_indices.unsqueeze(1)
        inside_mask = inside_mask & ~source_mask

        # Calculate areas only for inside points
        areas = torch.zeros_like(inside_mask, dtype=torch.float32)
        where_inside = torch.where(inside_mask)

        if where_inside[0].numel() > 0:
            # Calculate areas only for points inside triangles
            relevant_v0 = v0[where_inside]
            relevant_v1 = v1[where_inside]
            relevant_v2 = v2[where_inside]

            # Calculate areas using cross product
            cross_prod = torch.linalg.cross(
                relevant_v1 - relevant_v0,
                relevant_v2 - relevant_v0
            )
            areas[where_inside] = 0.5 * torch.norm(cross_prod, dim=1)

            del cross_prod, relevant_v0, relevant_v1, relevant_v2

        del v0, v1, v2, inside_mask, source_mask

        # Sum up the overlap penalty
        overlap_penalty = areas.sum()

        return overlap_penalty
