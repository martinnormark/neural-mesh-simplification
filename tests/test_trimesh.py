import trimesh
import pytest


@pytest.mark.skip(reason="Skipping trimesh tests. Run with trimesh mark to run.")
@pytest.mark.trimesh
class TestMeshCreation:
    def test_sphere(self):
        print("Creating sphere")
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=2)
        mesh.export("./tests/mesh_data/sphere_2.obj")

    def test_cube(self):
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        mesh.export("./tests/mesh_data/cube_2.obj")
