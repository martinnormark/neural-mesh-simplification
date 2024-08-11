import trimesh
import numpy as np


def hausdorff_distance(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, samples: int = 10000
):
    """
    Calculate the Hausdorff distance between two meshes.

    Parameters:
    mesh1 (trimesh.Trimesh): The first input mesh.
    mesh2 (trimesh.Trimesh): The second input mesh.
    samples (int): The number of samples to use for the calculation.

    Returns:
    float: The Hausdorff distance between the two meshes.
    """
    # Sample points from both meshes
    points1 = mesh1.sample(samples)
    points2 = mesh2.sample(samples)

    # Calculate distances from points1 to mesh2
    _, distances1, _ = trimesh.proximity.closest_point(mesh2, points1)

    # Calculate distances from points2 to mesh1
    _, distances2, _ = trimesh.proximity.closest_point(mesh1, points2)

    # Hausdorff distance is the maximum of all minimum distances
    return max(np.max(distances1), np.max(distances2))
