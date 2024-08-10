import trimesh
import numpy as np

def normal_consistency(mesh):
    """
    Calculate the normal vector consistency of a mesh.
    
    Parameters:
    mesh (trimesh.Trimesh): The input mesh.
    
    Returns:
    float: The normal vector consistency metric.
    """
    # Calculate face normals
    face_normals = mesh.face_normals
    
    # Calculate the dot product of adjacent face normals
    face_adjacency = mesh.face_adjacency
    normal_dots = np.einsum('ij,ij->i', face_normals[face_adjacency[:, 0]], face_normals[face_adjacency[:, 1]])
    
    # Calculate the consistency as the mean of the dot products
    consistency = np.mean(normal_dots)
    
    return consistency
