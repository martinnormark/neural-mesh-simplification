import numpy as np
import networkx as nx
from utils.mesh_operations import build_graph_from_mesh


def test_build_graph_from_mesh(sample_mesh):
    graph = build_graph_from_mesh(sample_mesh)

    # Check number of nodes and edges
    assert len(graph.nodes) == len(sample_mesh.vertices)
    assert len(graph.edges) == (
        3 * len(sample_mesh.faces) - len(sample_mesh.edges_unique)
    )

    # Check node attributes
    for i, pos in enumerate(sample_mesh.vertices):
        assert i in graph.nodes
        assert np.allclose(graph.nodes[i]["pos"], pos)

    # Check edge connectivity
    for face in sample_mesh.faces:
        assert graph.has_edge(face[0], face[1])
        assert graph.has_edge(face[1], face[2])
        assert graph.has_edge(face[2], face[0])

    # Check graph connectivity
    assert nx.is_connected(graph)
