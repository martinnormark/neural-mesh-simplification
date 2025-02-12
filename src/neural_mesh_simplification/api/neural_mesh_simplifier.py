import torch
import trimesh

from ..data.dataset import preprocess_mesh, mesh_to_dgl, dgl_to_trimesh
from ..models import NeuralMeshSimplification


class NeuralMeshSimplifier:
    def __init__(
        self,
        input_dim,
        hidden_dim,
        edge_hidden_dim,  # Separate hidden dim for edge predictor
        num_layers,
        k,
        edge_k,
        target_ratio
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_layers = num_layers
        self.k = k
        self.edge_k = edge_k
        self.target_ratio = target_ratio
        self.model = self._build_model()

    @classmethod
    def using_model(cls, at_path: str, hidden_dim: int, edge_hidden_dim: int, map_location: str):
        instance = cls(
            input_dim=3,
            hidden_dim=hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_layers=3,
            k=15,
            edge_k=15,
            target_ratio=0.5
        )
        instance._load_model(at_path, map_location)
        return instance

    def _build_model(self):
        return NeuralMeshSimplification(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            edge_hidden_dim=self.edge_hidden_dim,
            num_layers=self.num_layers,
            k=self.k,
            edge_k=self.edge_k,
            target_ratio=self.target_ratio
        )

    def _load_model(self, checkpoint_path: str, map_location: str):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    def simplify(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        # Preprocess the mesh (e.g. normalize, center)
        preprocessed_mesh: trimesh.Trimesh = preprocess_mesh(mesh)

        # Convert to a tensor
        graph, _ = mesh_to_dgl(preprocessed_mesh)
        s_graph, s_faces, _ = self.model(graph)

        return dgl_to_trimesh(s_graph, s_faces)
