import logging

import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph

from .edge_predictor import EdgePredictorDGL
from .face_classifier import FaceClassifierDGL
from .point_sampler import PointSamplerDGL

logger = logging.getLogger(__name__)


class NeuralMeshSimplification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_hidden_dim: int,
        num_layers: int,
        k: int,
        edge_k: int,
        target_ratio: float,
    ):
        super(NeuralMeshSimplification, self).__init__()
        self.k = k
        self.target_ratio = target_ratio

        self.point_sampler = PointSamplerDGL(input_dim, hidden_dim, num_layers)
        self.edge_predictor = EdgePredictorDGL(input_dim, edge_hidden_dim, edge_k)
        self.face_classifier = FaceClassifierDGL(input_dim, hidden_dim, num_layers, k)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
        """
        Forward pass for NeuralMeshSimplification.

        Args:
            g (dlg.DGLGraph): Input graph containing node features `x` and optionally positions `pos`.

        Returns:
            dlg.DGLGraph: The graph containing the simplified mesh
            torch.Tensor: The simplified faces
            torch.Tensor: The face probabilities from the Face Classifier
        """

        device = g.device

        logger.debug(f"Executing Mesh Simplification Forward pass on device {device}")

        x = g.ndata['x']
        pos = g.ndata['pos'] if 'pos' in g.ndata else x

        # Step 1: Sample points using the PointSamplerDGL
        logger.debug(f"Calling Point Sampler")
        sampled_indices, sampled_probs = self.sample_points(g)
        logger.debug(f"devices (sampled_indices, sampled_probs) = "
                     f"({sampled_indices.device}, {sampled_probs.device})")

        # Extract sampled features and positions
        sampled_x = x[sampled_indices]
        sampled_pos = pos[sampled_indices]

        # Create a new subgraph with sampled nodes
        logger.debug(f"Creating node subgraph with sampled nodes")
        sampled_g = dgl.node_subgraph(g, sampled_indices)
        logger.debug(f"devices sampled_g {sampled_g.device}")

        # Step 2: Predict edges using EdgePredictorDGL
        logger.debug(f"Calling Edge Predictor")
        edge_index_pred, edge_probs = self.edge_predictor(sampled_g)
        logger.debug(f"devices (edge_index_pred, edge_probs) = "
                     f"({edge_index_pred.device}, {edge_probs.device})")

        # Filter edges to keep only those connecting existing nodes
        # valid_edges = ((edge_index_pred[0] < sampled_indices.shape[0])
        #                & (edge_index_pred[1] < sampled_indices.shape[0]))
        # edge_index_pred = edge_index_pred[:, valid_edges]
        # edge_probs = edge_probs[valid_edges]

        # Step 3: Generate candidate triangles
        logger.debug(f"Generating candidate triangles")
        candidate_triangles, triangle_probs = self.generate_candidate_triangles(
            sampled_g,
            edge_probs
        )
        logger.debug(f"devices (candidate_triangles, triangle_probs) = "
                     f"({candidate_triangles.device}, {triangle_probs.device})")

        # Step 4: Classify faces using FaceClassifierDGL
        if candidate_triangles.shape[0] > 0:
            # Create features and positions for triangles
            triangle_features = sampled_x[candidate_triangles].mean(dim=1)
            triangle_centers = sampled_pos[candidate_triangles].mean(dim=1)

            # Create a new DGL graph for the triangles
            triangle_g = dgl.graph(
                ([], []),
                num_nodes=candidate_triangles.shape[0],
                device=device
            )
            logger.debug(f"Created a new DGLGraph for the triangles on device {triangle_g.device}")
            triangle_g.ndata['x'] = triangle_features
            triangle_g.ndata['pos'] = triangle_centers

            # Classify faces
            logger.debug(f"Calling Face Classifier")
            face_probs = self.face_classifier(triangle_g, triangle_centers)
            logger.debug(f"devices face_probs {face_probs.device}")
        else:
            face_probs = torch.empty(0, device=device)

        # Step 5: Filter triangles based on face probabilities
        if candidate_triangles.shape[0] == 0:
            simplified_faces = torch.empty(
                (0, 3), dtype=torch.long, device=device
            )
        else:
            threshold = torch.quantile(
                face_probs, 1 - self.target_ratio
            )  # Use a dynamic threshold
            simplified_faces = candidate_triangles[face_probs > threshold]

        # Create a new DGLGraph for the simplified mesh
        simplified_g = dgl.graph(
            (edge_index_pred[0], edge_index_pred[1]),
            num_nodes=sampled_indices.shape[0],
            device=device
        )
        logger.debug(f"Created a new DGLGraph for the simplified mesh on device {simplified_g.device}")

        # Ensure all sampled vertices are included
        all_nodes = torch.arange(sampled_indices.shape[0], device=device)
        simplified_g = dgl.add_self_loop(simplified_g)
        simplified_g = dgl.add_edges(simplified_g, all_nodes, all_nodes)

        logger.debug(f"devices (sampled_pos, sampled_x, sampled_probs) = "
                     f"({sampled_pos.device}, {sampled_x.device}, {sampled_probs.device})")
        simplified_g.ndata['pos'] = sampled_pos
        simplified_g.ndata['x'] = sampled_x
        simplified_g.ndata['sampled_prob'] = sampled_probs

        return simplified_g, simplified_faces, face_probs

    def sample_points(self, g: DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points using the PointSamplerDGL module.

        Args:
            g (DGLGraph): Input graph.

        Returns:
            tuple: Sampled indices and their probabilities.
        """
        num_nodes = g.num_nodes()

        # Determine the target number of nodes to sample
        target_nodes = min(max(int(self.target_ratio * num_nodes), 1), num_nodes)

        # Get sampling probabilities from PointSamplerDGL
        sampled_probs = self.point_sampler(g)

        # Select top-k nodes based on probabilities
        sampled_indices = torch.topk(sampled_probs, k=target_nodes).indices

        return sampled_indices, sampled_probs[sampled_indices]

    def generate_candidate_triangles(
        self,
        g: DGLGraph,
        edge_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate triangles from edges.

        Args:
            g (DGLGraph): Input graph with predicted edges.
            edge_probs (torch.Tensor): Probabilities of edges in the graph.

        Returns:
            tuple: Candidate triangles and their probabilities.
        """

        device = g.device

        edge_index = torch.stack(g.edges())

        # Handle the case when edge_index is empty
        if edge_index.numel() == 0:
            return (
                torch.empty((0, 3), dtype=torch.long, device=device),
                torch.empty(0, device=device)
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
