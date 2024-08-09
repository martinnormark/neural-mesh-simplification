import os
import sys

# Add the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MeshSimplificationDataset, load_mesh, preprocess_mesh
import trimesh
import networkx as nx
from tqdm import tqdm


def preprocess_dataset(
    input_dir,
    output_dir,
    pre_process=True,
    min_faces=1000,
    max_faces=50000,
    min_components=1,
    max_components=1,
    print_stats=False,
):
    dataset = MeshSimplificationDataset(data_dir=input_dir)

    for idx in tqdm(range(len(dataset)), desc="Processing meshes"):
        file_path = os.path.join(dataset.data_dir, dataset.file_list[idx])

        mesh = load_mesh(file_path)

        if pre_process:
            mesh = preprocess_mesh(mesh)

        if mesh is not None:
            face_adjacency = trimesh.graph.face_adjacency(mesh.faces)

            G = nx.Graph()
            G.add_edges_from(face_adjacency)

            components = list(nx.connected_components(G))

            num_components = len(components)
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)

            if num_components < min_components or num_components > max_components:
                print(f"Skipping mesh {idx}: {dataset.file_list[idx]}")
                print(f"  Connected components: {num_components}")
                print()
                continue

            if print_stats:
                print(f"Mesh {idx}: {dataset.file_list[idx]}")
                print(f"  Connected components: {num_components}")
                print(f"  Vertices: {num_vertices}")
                print(f"  Faces: {num_faces}")
                print(f"  Is watertight: {mesh.is_watertight}")
                print(f"  Volume: {mesh.volume}")
                print(f"  Surface area: {mesh.area}")

                non_manifold_edges = mesh.edges_unique[mesh.edges_unique_length > 2]
                print(f"  Number of non-manifold edges: {len(non_manifold_edges)}")
                print()

            output_file = os.path.join(output_dir, dataset.file_list[idx])
            mesh.export(output_file.replace(".ply", ".stl"))
        else:
            print(f"Failed to load mesh {idx}: {dataset.file_list[idx]}")
            print()

    print("Finished processing all meshes.")


if __name__ == "__main__":
    preprocess_dataset("data/raw", "data/processed")
