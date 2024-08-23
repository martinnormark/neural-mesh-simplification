import torch
import torch.nn as nn
from torch_cluster import knn


class EdgeCrossingLoss(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k  # Number of nearest triangles to consider

    def forward(self, simplified_data) -> torch.Tensor:
        vertices = simplified_data["sampled_vertices"]
        faces = simplified_data["simplified_faces"]
        face_probs = simplified_data["face_probs"]

        # 1. Find k-nearest triangles for each triangle
        nearest_triangles = self.find_nearest_triangles(vertices, faces)

        # 2. Detect edge crossings between nearby triangles
        crossings = self.detect_edge_crossings(vertices, faces, nearest_triangles)

        # 3. Calculate loss
        loss = self.calculate_loss(crossings, face_probs)

        return loss

    def find_nearest_triangles(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        # Compute triangle centroids
        centroids = vertices[faces].mean(dim=1)

        # Use knn to find nearest triangles
        k = min(
            self.k, centroids.shape[0]
        )  # Ensure k is not larger than the number of centroids
        _, indices = knn(centroids, centroids, k=k)

        # Reshape indices to [num_faces, k]
        indices = indices.view(centroids.shape[0], k)

        # Remove self-connections (triangles cannot be their own neighbor)
        nearest = []
        for i in range(indices.shape[0]):
            neighbors = indices[i][indices[i] != i]
            if len(neighbors) == 0:
                nearest.append(torch.empty(0, dtype=torch.long))
            else:
                nearest.append(neighbors[: self.k - 1])

        # Return tensor with consistent shape
        if len(nearest) > 0 and all(len(n) == 0 for n in nearest):
            nearest = torch.empty((len(nearest), 0), dtype=torch.long)
        else:
            nearest = torch.stack(nearest)
        return nearest

    def detect_edge_crossings(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        nearest_triangles: torch.Tensor,
    ) -> torch.Tensor:
        def edges_of_triangle(triangle):
            # Extracts the edges from a triangle defined by vertex indices
            return [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ]

        def edge_crosses(edge1, edge2):
            def vector(p1, p2):
                return p2 - p1

            def cross_product(v1, v2):
                return torch.cross(v1, v2, dim=-1)

            def dot_product(v1, v2):
                return torch.dot(v1, v2)

            def is_between(p, edge):
                return torch.all(
                    torch.abs(p - edge[0]) + torch.abs(p - edge[1])
                    == torch.abs(edge[1] - edge[0])
                )

            # Edge1: (A, B), Edge2: (C, D)
            A, B = vertices[edge1[0]], vertices[edge1[1]]
            C, D = vertices[edge2[0]], vertices[edge2[1]]
            AB = vector(A, B)
            CD = vector(C, D)
            AC = vector(A, C)
            CA = vector(C, A)

            # Check if lines are co-planar
            if dot_product(cross_product(AB, AC), CD) == 0:
                return False

            # Check if edges are crossing
            denom = dot_product(cross_product(AB, CD), CD)
            if denom == 0:
                return False

            num = dot_product(cross_product(CA, CD), CD)
            t = num / denom

            if 0 <= t <= 1:
                intersection = A + t * AB
                return is_between(intersection, (C, D)) and is_between(
                    intersection, (A, B)
                )

            return False

        crossings = torch.zeros(
            faces.shape[0], dtype=torch.float, device=vertices.device
        )
        for i, face in enumerate(faces):
            for j in nearest_triangles[i]:
                if j == -1:
                    continue
                for edge1 in edges_of_triangle(face):
                    for edge2 in edges_of_triangle(faces[j]):
                        if edge_crosses(edge1, edge2):
                            crossings[i] += 1
        return crossings

    def calculate_loss(
        self, crossings: torch.Tensor, face_probs: torch.Tensor
    ) -> torch.Tensor:
        # Weighted sum of crossings by triangle probabilities
        loss = torch.sum(face_probs * crossings)
        return loss
