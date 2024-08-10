import trimesh
import numpy as np


def chamfer_distance(mesh1, mesh2, samples=10000):
    points1 = mesh1.sample(samples)
    points2 = mesh2.sample(samples)

    _, distances1, _ = trimesh.proximity.closest_point(mesh2, points1)
    _, distances2, _ = trimesh.proximity.closest_point(mesh1, points2)

    chamfer_dist = np.mean(distances1) + np.mean(distances2)
    return chamfer_dist
