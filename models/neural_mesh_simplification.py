import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from models import PointSampler, EdgePredictor, FaceClassifier


class NeuralMeshSimplification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, k=20, edge_k=None):
        super(NeuralMeshSimplification, self).__init__()
        self.point_sampler = PointSampler(input_dim, hidden_dim)
        self.edge_predictor = EdgePredictor(
            input_dim, hidden_dim, k=edge_k if edge_k is not None else k
        )
        self.face_classifier = FaceClassifier(input_dim, hidden_dim, num_layers, k)
        self.k = k

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        # Sample points
        sampled_probs = self.point_sampler(x, edge_index)
        num_samples = min(max(int(0.5 * num_nodes), 1), num_nodes - 1)
        sampled_indices = torch.multinomial(
            sampled_probs, num_samples=num_samples, replacement=False
        )
        sampled_indices = torch.clamp(sampled_indices, 0, num_nodes - 1)
        sampled_indices = torch.unique(sampled_indices)

        sampled_x = x[sampled_indices]
        sampled_pos = (
            data.pos[sampled_indices]
            if hasattr(data, "pos") and data.pos is not None
            else sampled_x
        )

        sampled_vertices = (
            data.pos[sampled_indices]
            if hasattr(data, "pos") and data.pos is not None
            else sampled_x
        )

        # Update edge_index to reflect the new indices
        sampled_edge_index, _ = torch_geometric.utils.subgraph(
            sampled_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )

        # Predict edges
        edge_index_pred, edge_probs = self.edge_predictor(sampled_x, sampled_edge_index)

        # Generate candidate triangles
        candidate_triangles, triangle_probs = self.generate_candidate_triangles(
            edge_index_pred, edge_probs
        )

        # Classify faces
        if candidate_triangles.shape[0] > 0:
            face_probs = self.face_classifier(sampled_x, sampled_pos, batch=None)
            # Ensure face_probs matches the number of candidate triangles
            face_probs = face_probs[: candidate_triangles.shape[0]]
        else:
            face_probs = torch.empty(0, device=data.x.device)

        if candidate_triangles.shape[0] == 0:
            simplified_faces = torch.empty(
                (0, 3), dtype=torch.long, device=data.x.device
            )
        else:
            simplified_faces = candidate_triangles[face_probs > 0.5]

        return {
            "sampled_indices": sampled_indices,
            "sampled_probs": sampled_probs,
            "sampled_vertices": sampled_vertices,
            "edge_index": edge_index_pred,
            "edge_probs": edge_probs,
            "candidate_triangles": candidate_triangles,
            "triangle_probs": triangle_probs,
            "face_probs": face_probs,
            "simplified_faces": simplified_faces,
        }

    def generate_candidate_triangles(self, edge_index, edge_probs):
        device = edge_index.device

        # Handle the case when edge_index is empty
        if edge_index.numel() == 0:
            return torch.empty((0, 3), dtype=torch.long, device=device), torch.empty(
                0, device=device
            )

        num_nodes = edge_index.max().item() + 1

        # Create an adjacency matrix from the edge index
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)

        # Check if edge_probs is a tuple or a tensor
        if isinstance(edge_probs, tuple):
            edge_indices, edge_values = edge_probs
            adj_matrix[edge_indices[0], edge_indices[1]] = edge_values
        else:
            adj_matrix[edge_index[0], edge_index[1]] = edge_probs

        # Adjust k based on the number of nodes
        k = min(self.k, num_nodes - 1)

        # Find k-nearest neighbors for each node
        _, knn_indices = torch.topk(adj_matrix, k=k, dim=1)

        # Generate candidate triangles
        triangles = []
        triangle_probs = []

        for i in range(num_nodes):
            neighbors = knn_indices[i]
            for j in range(k):
                for l in range(j + 1, k):
                    n1, n2 = neighbors[j], neighbors[l]
                    if adj_matrix[n1, n2] > 0:  # Check if the third edge exists
                        triangle = torch.tensor([i, n1, n2], device=device)
                        triangles.append(triangle)

                        # Calculate triangle probability
                        prob = (
                            adj_matrix[i, n1] * adj_matrix[i, n2] * adj_matrix[n1, n2]
                        ) ** (1 / 3)
                        triangle_probs.append(prob)

        if triangles:
            triangles = torch.stack(triangles)
            triangle_probs = torch.tensor(triangle_probs, device=device)
        else:
            triangles = torch.empty((0, 3), dtype=torch.long, device=device)
            triangle_probs = torch.empty(0, device=device)

        return triangles, triangle_probs
