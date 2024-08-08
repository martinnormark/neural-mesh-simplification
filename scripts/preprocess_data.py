import os
from data.dataset import load_mesh, preprocess_mesh
from utils.mesh_operations import simplify_mesh


def preprocess_dataset(input_dir, output_dir, target_faces):
    # Iterate through all meshes in input_dir
    # Load, preprocess, simplify, and save to output_dir
    pass


if __name__ == "__main__":
    preprocess_dataset("data/raw", "data/processed", target_faces=1000)
