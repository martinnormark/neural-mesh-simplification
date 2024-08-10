import trimesh
import numpy as np


def normal_consistency(mesh, samples=10000):
    """
    Calculate the normal vector consistency of a mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    samples (int): The number of samples to use for the calculation.

    Returns:
    float: The normal vector consistency metric.
    """
    points = mesh.sample(samples)

    _, _, face_indices = trimesh.proximity.closest_point(mesh, points)
    face_normals = mesh.face_normals[face_indices]

    mesh.vertex_normals

    closest_vertices = mesh.nearest.vertex(points)[1]
    vertex_normals = mesh.vertex_normals[closest_vertices]

    consistency = np.abs(np.sum(vertex_normals * face_normals, axis=1))

    return np.mean(consistency)
