import trimesh
import numpy as np

def edge_preservation(original_mesh, simplified_mesh):
    """
    Calculate the edge preservation metric between the original and simplified meshes.

    Parameters:
    original_mesh (trimesh.Trimesh): The original high-resolution mesh.
    simplified_mesh (trimesh.Trimesh): The simplified mesh.

    Returns:
    float: The edge preservation metric.
    """
    original_edges = set(map(tuple, map(sorted, original_mesh.edges_unique)))
    simplified_edges = set(map(tuple, map(sorted, simplified_mesh.edges_unique)))

    preserved_edges = original_edges.intersection(simplified_edges)
    edge_preservation_metric = len(preserved_edges) / len(original_edges)

    return edge_preservation_metric
