import logging

import dgl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TriangleCollisionLoss(nn.Module):
    def __init__(
        self, epsilon=1e-8, k=20, collision_threshold=1e-10, normal_threshold=0.99
    ):
        super().__init__()
        self.epsilon = epsilon
        self.k = k
        self.collision_threshold = collision_threshold
        self.normal_threshold = normal_threshold

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        face_probabilities: torch.Tensor
    ) -> torch.Tensor:
        logger.debug(f"Calculating TRIANGLE COLLISION loss")
        logger.debug(
            f"devices (vertices, faces, face_probabilities) = "
            f"({vertices}, {faces}, {face_probabilities})"
        )

        num_faces = faces.shape[0]

        if num_faces == 0:
            return torch.tensor(0.0, device=vertices.device)

        # Ensure face_probabilities matches the number of faces
        face_probabilities = torch.nn.functional.pad(
            face_probabilities, (0, max(0, num_faces - face_probabilities.shape[0]))
        )[:num_faces]

        v0, v1, v2 = vertices[faces].unbind(1)

        # Calculate face normals
        edges1 = v1 - v0
        edges2 = v2 - v0
        face_normals = torch.linalg.cross(edges1, edges2)
        face_normal_lengths = torch.norm(face_normals, dim=1, keepdim=True)
        face_normals = face_normals / (face_normal_lengths + self.epsilon)

        del edges1, edges2, face_normal_lengths  # No longer needed

        # Calculate centroids
        centroids = (v0 + v1 + v2) / 3

        # Find k nearest neighbors using DGL
        g_knn = dgl.knn_graph(centroids, self.k)
        neighbors = g_knn.edges()[1].reshape(-1, self.k)

        collision_count = torch.zeros(num_faces, device=vertices.device)
        for i in range(num_faces):
            nearby_faces = neighbors[i]
            nearby_v0, nearby_v1, nearby_v2 = (
                v0[nearby_faces],
                v1[nearby_faces],
                v2[nearby_faces],
            )
            collisions = self.check_triangle_intersection(
                v0[i],
                v1[i],
                v2[i],
                face_normals[i],
                nearby_v0,
                nearby_v1,
                nearby_v2,
                face_normals[nearby_faces],
                faces[i],
                faces[nearby_faces],
                centroids[i],
                centroids[nearby_faces],
            )
            collision_count[i] += collisions.sum()

        total_loss = torch.sum(face_probabilities * collision_count)
        return total_loss

    def check_triangle_intersection(
        self,
        v0,
        v1,
        v2,
        normal,
        nearby_v0,
        nearby_v1,
        nearby_v2,
        nearby_normals,
        face,
        nearby_faces,
        centroid,
        nearby_centroids,
    ):
        # Check if triangles are coplanar (relaxed condition)
        normal_dot = torch.abs(torch.sum(normal * nearby_normals, dim=1))
        coplanar = normal_dot > self.normal_threshold

        # Check for triangle-triangle intersections
        intersections = torch.zeros(
            len(nearby_faces), dtype=torch.bool, device=v0.device
        )

        for i, (nv0, nv1, nv2) in enumerate(zip(nearby_v0, nearby_v1, nearby_v2)):
            if coplanar[i]:
                # For coplanar triangles, check distance between centroids
                dist = torch.norm(centroid - nearby_centroids[i])
                intersections[i] = dist < self.collision_threshold
            else:
                intersections[i] = self.check_triangle_triangle_intersection(
                    v0, v1, v2, nv0, nv1, nv2
                )

        # Check if triangles are adjacent
        adjacent = torch.tensor(
            [len(set(face.tolist()) & set(nf.tolist())) >= 2 for nf in nearby_faces],
            dtype=torch.bool,
            device=v0.device
        )

        collisions = intersections & ~adjacent

        return collisions

    def check_triangle_triangle_intersection(self, v0, v1, v2, w0, w1, w2):
        def triangle_plane_intersection(t0, t1, t2, p0, p1, p2):
            n = torch.linalg.cross(t1 - t0, t2 - t0)
            d0, d1, d2 = (
                self.dist_dot(p0, t0, n),
                self.dist_dot(p1, t0, n),
                self.dist_dot(p2, t0, n),
            )
            return (d0 * d1 <= 0) or (d0 * d2 <= 0) or (d1 * d2 <= 0)

        return triangle_plane_intersection(
            v0, v1, v2, w0, w1, w2
        ) and triangle_plane_intersection(w0, w1, w2, v0, v1, v2)

    @staticmethod
    def dist_dot(p, t0, n):
        return torch.dot(p - t0, n)
