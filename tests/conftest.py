import pytest
import trimesh


@pytest.fixture
def sample_mesh():
    # Create a simple cube mesh for testing
    mesh = trimesh.creation.box()
    return mesh
