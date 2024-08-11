import numpy as np
from scipy.spatial import cKDTree


def calculate_dihedral_angles(mesh):
    """Calculate dihedral angles for all edges in the mesh."""
    face_adjacency = mesh.face_adjacency
    face_adjacency_angles = mesh.face_adjacency_angles

    edge_to_angle = {}
    for (face1, face2), angle in zip(face_adjacency, face_adjacency_angles):
        edge = tuple(sorted(set(mesh.faces[face1]) & set(mesh.faces[face2])))
        edge_to_angle[edge] = angle

    dihedral_angles = np.zeros(len(mesh.edges))
    for i, edge in enumerate(mesh.edges):
        edge = tuple(sorted(edge))
        dihedral_angles[i] = edge_to_angle.get(edge, 0)  # 0 for boundary edges

    return dihedral_angles


def edge_preservation(
    original_mesh, simplified_mesh, angle_threshold=30, important_edge_factor=2.0
):
    """
    Calculate the edge preservation metric between the original and simplified meshes.

    Parameters:
    original_mesh (trimesh.Trimesh): The original high-resolution mesh.
    simplified_mesh (trimesh.Trimesh): The simplified mesh.
    angle_threshold (float): The dihedral angle threshold for important edges (in degrees).
    important_edge_factor (float): Factor to increase weight for important edges.

    Returns:
    float: The edge preservation metric.
    """
    original_dihedral = calculate_dihedral_angles(original_mesh)
    important_original = original_dihedral > np.radians(angle_threshold)

    # Calculate edge midpoints
    original_midpoints = original_mesh.vertices[original_mesh.edges].mean(axis=1)
    simplified_midpoints = simplified_mesh.vertices[simplified_mesh.edges].mean(axis=1)

    tree = cKDTree(simplified_midpoints)

    # Find closest simplified edge for each original edge
    distances, _ = tree.query(original_midpoints)

    # Calculate weighted preservation
    weights = np.exp(original_dihedral)
    weights[important_original] *= important_edge_factor
    weighted_distances = distances * weights

    max_distance = np.max(original_mesh.bounds) - np.min(original_mesh.bounds)
    preservation_metric = 1 - np.mean(weighted_distances) / max_distance

    return preservation_metric
