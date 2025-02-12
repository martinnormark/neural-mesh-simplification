import logging

import dgl
import torch
import torch.nn as nn

from .chamfer_distance_loss import ProbabilisticChamferDistanceLoss
from .edge_crossing_loss import EdgeCrossingLoss
from .overlapping_triangles_loss import OverlappingTrianglesLoss
from .surface_distance_loss import ProbabilisticSurfaceDistanceLoss
from .triangle_collision_loss import TriangleCollisionLoss

logger = logging.getLogger(__name__)


class CombinedMeshSimplificationLoss(nn.Module):
    def __init__(
            self,
            lambda_c: float = 1.0,
            lambda_e: float = 1.0,
            lambda_o: float = 1.0
    ):
        super().__init__()
        self.prob_chamfer_loss = ProbabilisticChamferDistanceLoss()
        self.prob_surface_loss = ProbabilisticSurfaceDistanceLoss()
        self.collision_loss = TriangleCollisionLoss()
        self.edge_crossing_loss = EdgeCrossingLoss()
        self.overlapping_triangles_loss = OverlappingTrianglesLoss()
        self.lambda_c = lambda_c
        self.lambda_e = lambda_e
        self.lambda_o = lambda_o

    def forward(
            self,
            original_graph: dgl.DGLGraph,
            original_faces: torch.Tensor,
            sampled_graph: dgl.DGLGraph,
            sampled_faces: torch.Tensor,
            face_probs: torch.Tensor
    ):
        logger.debug(f"Calculating combined loss on device {original_graph.device}")

        orig_vertices = original_graph.ndata['x']
        sampled_vertices = sampled_graph.ndata['x']
        sampled_probs = sampled_graph.ndata['sampled_prob']

        del original_graph

        chamfer_loss = self.prob_chamfer_loss(
            orig_vertices,
            sampled_vertices,
            sampled_probs
        )

        del sampled_probs

        surface_loss = self.prob_surface_loss(
            orig_vertices,
            original_faces,
            sampled_vertices,
            sampled_faces,
            face_probs
        )

        del original_faces

        collision_loss = self.collision_loss(
            sampled_vertices,
            sampled_faces,
            face_probs
        )
        edge_crossing_loss = self.edge_crossing_loss(
            sampled_vertices,
            sampled_faces,
            face_probs
        )

        del face_probs

        overlapping_triangles_loss = self.overlapping_triangles_loss(sampled_vertices, sampled_faces)

        total_loss = (
                chamfer_loss
                + surface_loss
                + self.lambda_c * collision_loss
                + self.lambda_e * edge_crossing_loss
                + self.lambda_o * overlapping_triangles_loss
        )

        return total_loss
