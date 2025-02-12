import logging

import dgl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EdgeCrossingLoss(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k  # Number of nearest triangles to consider

    def forward(
            self,
            vertices: torch.Tensor,
            faces: torch.Tensor,
            face_probs: torch.Tensor
    ) -> torch.Tensor:

        logger.debug(f"Calculating EDGE CROSSING loss")
        logger.debug(
            f"devices (vertices, faces, face_probs) = "
            f"({vertices}, {faces}, {face_probs})"
        )

        # If no faces, return zero loss
        if faces.shape[0] == 0:
            return torch.tensor(0.0, device=vertices.device)
        # Ensure face_probs matches the number of faces
        if face_probs.shape[0] > faces.shape[0]:
            face_probs = face_probs[: faces.shape[0]]
        elif face_probs.shape[0] < faces.shape[0]:
            # Pad with zeros if we have fewer probabilities than faces
            padding = torch.zeros(
                faces.shape[0] - face_probs.shape[0], device=face_probs.device
            )
            face_probs = torch.cat([face_probs, padding])

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
        g_knn = dgl.knn_graph(centroids, k, exclude_self=True)

        return g_knn.edges()[1].reshape(-1, k)

    def detect_edge_crossings(
            self,
            vertices: torch.Tensor,
            faces: torch.Tensor,
            nearest_triangles: torch.Tensor,
    ) -> torch.Tensor:
        def edge_vectors(triangles):
            # Extracts the edges from a triangle defined by vertex indices
            return vertices[triangles[:, [1, 2, 0]]] - vertices[triangles]

        edges = edge_vectors(faces)
        crossings = torch.zeros(faces.shape[0], device=vertices.device)

        for i in range(faces.shape[0]):
            neighbor_edges = edge_vectors(faces[nearest_triangles[i]])
            for j in range(3):
                edge = edges[i, j].unsqueeze(0).unsqueeze(0)
                cross_product = torch.cross(edge.expand(neighbor_edges.shape), neighbor_edges, dim=-1)
                t = torch.sum(cross_product * neighbor_edges, dim=-1) / torch.sum(
                    cross_product * edge.expand(neighbor_edges.shape), dim=-1)
                u = torch.sum(cross_product * edges[i].unsqueeze(0), dim=-1) / torch.sum(
                    cross_product * edge.expand(neighbor_edges.shape), dim=-1)
                mask = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
                crossings[i] += mask.sum()

        return crossings

    def calculate_loss(
            self, crossings: torch.Tensor, face_probs: torch.Tensor
    ) -> torch.Tensor:
        # Weighted sum of crossings by triangle probabilities
        num_faces = face_probs.shape[0]
        return torch.sum(face_probs * crossings, dtype=torch.float32) / num_faces
