import os
import torch
import trimesh
import numpy as np
from typing import Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.mesh_operations import build_graph_from_mesh


class MeshSimplificationDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        return [
            f
            for f in os.listdir(self.data_dir)
            if f.endswith(".ply") or f.endswith(".obj") or f.endswith(".stl")
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        mesh = load_mesh(file_path)
        mesh = preprocess_mesh(mesh)

        if self.transform:
            mesh = self.transform(mesh)

        data = mesh_to_tensor(mesh)
        return data


def load_mesh(file_path: str) -> trimesh.Trimesh:
    """Load a mesh from file."""
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def preprocess_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Preprocess a mesh (e.g., normalize, center)."""
    if mesh is None:
        return None

    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Scale to unit cube
    max_dim = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices /= max_dim

    return mesh


def augment_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply data augmentation to a mesh."""
    if mesh is None:
        return None

    # Example: Random rotation
    rotation = trimesh.transformations.random_rotation_matrix()
    mesh.apply_transform(rotation)

    return mesh


def mesh_to_tensor(mesh: trimesh.Trimesh) -> Data:
    """Convert a mesh to tensor representation including graph structure."""
    if mesh is None:
        return None

    # Convert vertices and faces to tensors
    vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.long)

    # Build graph structure
    G = build_graph_from_mesh(mesh)

    # Create edge index tensor
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    # Create Data object
    data = Data(
        x=vertices_tensor,
        edge_index=edge_index,
        face=faces_tensor.t(),
        num_nodes=len(mesh.vertices),
    )

    return data
