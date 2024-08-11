import trimesh
import numpy as np


def chamfer_distance(
    mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, samples: int = 10000
):
    """
    Calculate the Chamfer distance between two meshes.

    Parameters:
    mesh1 (trimesh.Trimesh): The first input mesh.
    mesh2 (trimesh.Trimesh): The second input mesh.
    samples (int): The number of samples to use for the calculation.

    Returns:
    float: The Chamfer distance metric
    """
    points1 = mesh1.sample(samples)
    points2 = mesh2.sample(samples)

    _, distances1, _ = trimesh.proximity.closest_point(mesh2, points1)
    _, distances2, _ = trimesh.proximity.closest_point(mesh1, points2)

    chamfer_dist = np.mean(distances1) + np.mean(distances2)

    return chamfer_dist
