import torch
import torch.nn as nn


class PointSampler(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, mesh_tensor):
        # Extract vertices and faces from the mesh tensor
        num_vertices = int(
            mesh_tensor[0].item()
        )  # Assuming the first element stores the number of vertices
        vertices = mesh_tensor[1 : num_vertices * 3 + 1].view(-1, 3)
        faces = mesh_tensor[num_vertices * 3 + 1 :].view(-1, 3).long()

        # Sample points
        sampled_points = self.sample_points_from_mesh(vertices, faces)
        return sampled_points

    def sample_points_from_mesh(self, vertices, faces):
        # Compute face areas
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_areas = 0.5 * torch.norm(
            torch.linalg.cross(v1 - v0, v2 - v0, dim=1), dim=1
        )

        # Normalize face areas to get probabilities
        face_probs = face_areas / face_areas.sum()

        # Sample faces based on their areas
        face_indices = torch.multinomial(face_probs, self.num_samples, replacement=True)

        # Sample points within the selected faces
        u = torch.sqrt(torch.rand(self.num_samples, device=vertices.device))
        v = torch.rand(self.num_samples, device=vertices.device)
        w = 1 - u - v

        sampled_faces = faces[face_indices]
        v0, v1, v2 = (
            vertices[sampled_faces[:, 0]],
            vertices[sampled_faces[:, 1]],
            vertices[sampled_faces[:, 2]],
        )

        sampled_points = w[:, None] * v0 + u[:, None] * v1 + v[:, None] * v2

        return sampled_points
