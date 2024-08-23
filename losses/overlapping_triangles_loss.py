import torch
import torch.nn as nn


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

    def forward(self, simplified_data):
        vertices = simplified_data["sampled_vertices"]
        faces = simplified_data["simplified_faces"]

        # 1. Sample points from each triangle
        sampled_points = self.sample_points_from_triangles(vertices, faces)

        # 2. Find k-nearest triangles for each point
        nearest_triangles = self.find_nearest_triangles(sampled_points, vertices, faces)

        # 3. Detect overlaps and calculate the loss
        overlap_penalty = self.calculate_overlap_loss(
            sampled_points, vertices, faces, nearest_triangles
        )

        return overlap_penalty

    def sample_points_from_triangles(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples points from each triangle in the mesh.

        Args:
            vertices (torch.Tensor): The vertex positions (V x 3).
            faces (torch.Tensor): The indices of the vertices that make up each triangle (F x 3).

        Returns:
            torch.Tensor: Sampled points (F * num_samples x 3).
        """
        # Get the vertices for each face
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Generate random barycentric coordinates
        u = torch.rand((faces.shape[0], self.num_samples, 1), device=vertices.device)
        v = torch.rand((faces.shape[0], self.num_samples, 1), device=vertices.device)

        # Ensure the points are inside the triangle
        condition = u + v > 1
        u[condition] = 1 - u[condition]
        v[condition] = 1 - v[condition]

        # Calculate the coordinates of the sampled points
        points = (
            v0.unsqueeze(1) * (1 - u - v) + v1.unsqueeze(1) * u + v2.unsqueeze(1) * v
        )

        # Reshape to a (F * num_samples x 3) tensor
        sampled_points = points.view(-1, 3)

        return sampled_points

    def find_nearest_triangles(
        self, sampled_points: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
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

        # Use knn to find nearest triangles for each sampled point
        _, indices = torch.cdist(sampled_points, centroids).topk(self.k, largest=False)

        return indices

    def calculate_overlap_loss(
        self,
        sampled_points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        nearest_triangles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the overlap loss by checking if sampled points belong to multiple triangles.

        Args:
            sampled_points (torch.Tensor): Sampled points from triangles (N x 3).
            vertices (torch.Tensor): The vertex positions (V x 3).
            faces (torch.Tensor): The indices of the vertices that make up each triangle (F x 3).
            nearest_triangles (torch.Tensor): Indices of the k-nearest triangles for each sampled point (N x k).

        Returns:
            torch.Tensor: The overlap penalty loss.
        """
        overlap_penalty = 0.0

        for i, point in enumerate(sampled_points):
            # Get the k-nearest triangles for this point
            triangles = faces[nearest_triangles[i]]

            # Calculate the area sum for point with each of the triangles
            for triangle in triangles:
                v0, v1, v2 = vertices[triangle]
                area = self.calculate_area(point, v0, v1, v2)
                if area > 0:  # If the point lies inside the triangle
                    overlap_penalty += (
                        area  # Increase penalty if point belongs to multiple triangles
                    )

        return overlap_penalty

    def calculate_area(
        self, point: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
    ) -> float:
        """
        Calculates the area of a triangle formed by the point and a triangle's vertices.

        Args:
            point (torch.Tensor): The point to check (3).
            v0, v1, v2 (torch.Tensor): The vertices of the triangle (3 each).

        Returns:
            float: The area of the triangle formed by the point and the triangle's vertices.
        """
        # Vector cross product to calculate area
        cross_prod = torch.cross(v1 - v0, v2 - v0)
        area = 0.5 * torch.norm(cross_prod)

        # Check if the point lies within the triangle
        # Barycentric coordinates or area comparison can be used
        if torch.abs(area) > 1e-6:  # Ensure it's a valid triangle
            u = torch.cross(v2 - v1, point - v1).dot(cross_prod)
            v = torch.cross(v0 - v2, point - v2).dot(cross_prod)
            w = torch.cross(v1 - v0, point - v0).dot(cross_prod)
            if u >= 0 and v >= 0 and w >= 0:
                return area

        return 0.0  # Return 0 if the point doesn't lie within the triangle
