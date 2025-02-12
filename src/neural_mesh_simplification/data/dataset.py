import logging
import os
from typing import Optional

import dgl
import numpy as np
import torch
import trimesh
from dgl.data import DGLDataset
from trimesh import Geometry, Trimesh

logger = logging.getLogger(__name__)


class MeshSimplificationDataset(DGLDataset):
    def __init__(self, data_dir, preprocess: bool = False, transform: Optional[callable] = None):
        super().__init__(name='mesh_simplification')
        self.data_dir = data_dir
        self.preprocess = preprocess
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

    def __getitem__(self, idx) -> tuple[dgl.DGLGraph, torch.Tensor]:
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        mesh = load_mesh(file_path)

        if self.preprocess:
            mesh = preprocess_mesh(mesh)

        if self.transform:
            mesh = self.transform(mesh)

        return mesh_to_dgl(mesh)


def load_mesh(file_path: str) -> Geometry | list[Geometry] | None:
    """Load a mesh from file."""
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh {file_path}: {e}")
        return None


def preprocess_mesh(mesh: trimesh.Trimesh) -> Trimesh | None:
    """Preprocess a mesh (e.g., normalize, center)."""
    if mesh is None:
        return None

    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Scale to unit cube
    max_dim = np.max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
    mesh.vertices /= max_dim

    return mesh


def augment_mesh(mesh: trimesh.Trimesh) -> Trimesh | None:
    """Apply data augmentation to a mesh."""
    if mesh is None:
        return None

    # Example: Random rotation
    rotation = trimesh.transformations.random_rotation_matrix()
    mesh.apply_transform(rotation)

    return mesh


def mesh_to_dgl(mesh) -> tuple[dgl.DGLGraph, torch.Tensor]:
    if mesh is None:
        raise ValueError("Mesh is undefined")

    # Convert vertices to tensor
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    num_nodes = vertices.shape[0]

    # Convert unique edges
    edges_np = np.array(list(mesh.edges_unique))
    edges = torch.tensor(edges_np, dtype=torch.long).t()

    # Create DGL graph
    g = dgl.graph((edges[0], edges[1]), num_nodes=num_nodes)
    g = dgl.add_self_loop(g)

    # Verify node count matches
    assert g.number_of_nodes() == vertices.shape[0], "Mismatch between nodes and features"

    # Add node features
    g.ndata['x'] = vertices
    g.ndata['pos'] = vertices

    # Store face information as node data
    if hasattr(mesh, 'faces'):
        faces_tensor = torch.tensor(mesh.faces, dtype=torch.long)
    else:
        faces_tensor = torch.empty((0, 3), dtype=torch.long)

    return g, faces_tensor


def dgl_to_trimesh(g: dgl.DGLGraph, faces: torch.Tensor | None) -> Trimesh:
    # Convert to a tensor
    vertices = g.ndata['pos'].numpy()
    vertex_normals = g.ndata.get('normal', None)
    if vertex_normals is not None:
        vertex_normals = vertex_normals.numpy()

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        process=True,
        validate=True
    )


def collate(batch: list[tuple]) -> tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, faces = zip(*batch)
    max_faces = max(f.shape[0] for f in faces)
    padded_faces = torch.stack([
        torch.nn.functional.pad(f, (0, 0, 0, max_faces - f.shape[0]), value=-1)
        for f in faces
    ])
    return graphs, padded_faces
