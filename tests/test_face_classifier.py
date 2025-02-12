import dgl
import pytest
import torch

from neural_mesh_simplification.models import FaceClassifierDGL


@pytest.fixture
def face_classifier():
    return FaceClassifierDGL(input_dim=16, hidden_dim=32, num_layers=3, k=20)


@pytest.fixture
def sample_graph() -> dgl.DGLGraph:
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=x.shape[0])
    g.ndata['x'] = x
    return g


def test_face_classifier_initialization(face_classifier):
    assert len(face_classifier.layers) == 3
    assert isinstance(face_classifier.final_layer, torch.nn.Linear)


@pytest.mark.skip(reason="Convert to DGL before re-enabling")
def test_face_classifier_forward(face_classifier, sample_graph):
    num_nodes = sample_graph.num_nodes()
    centers = torch.randn(num_nodes, 3)

    out = face_classifier(sample_graph, centers)
    assert out.shape == (centers,)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.skip(reason="Convert to DGL before re-enabling")
def test_face_classifier_gradient(face_classifier):
    num_faces = 100
    x = torch.randn(num_faces, 16, requires_grad=True)
    pos = torch.randn(num_faces, 3, requires_grad=True)

    out = face_classifier(x, pos)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert pos.grad is not None
    assert all(p.grad is not None for p in face_classifier.parameters())


@pytest.mark.skip(reason="Convert to DGL before re-enabling")
def test_face_classifier_with_batch(face_classifier):
    num_faces = 100
    batch_size = 2
    x = torch.randn(num_faces, 16)
    pos = torch.randn(num_faces, 3)
    batch = torch.cat(
        [torch.full((num_faces // batch_size,), i) for i in range(batch_size)]
    )

    out = face_classifier(x, pos, batch)
    assert out.shape == (num_faces,)
    assert torch.all(out >= 0) and torch.all(out <= 1)

    # Check if the sum of probabilities for each batch is close to 1
    for i in range(batch_size):
        batch_sum = out[batch == i].sum()
        assert torch.isclose(batch_sum, torch.tensor(1.0), atol=1e-6)


@pytest.mark.skip(reason="Convert to DGL before re-enabling")
def test_face_classifier_knn_graph(face_classifier):
    num_faces = 100
    x = torch.randn(num_faces, 16)
    pos = torch.randn(num_faces, 3, 3)  # 3 vertices per face

    # Call the forward method to construct the k-nn graph
    _ = face_classifier(x, pos)

    # Get the constructed edge_index from the first TriConv layer
    edge_index = face_classifier.triconv_layers[0].last_edge_index

    # Check the number of neighbors for each face
    for i in range(num_faces):
        actual_neighbors = edge_index[1][edge_index[0] == i]
        assert (
            len(actual_neighbors) >= face_classifier.k
        ), f"Face {i} has {len(actual_neighbors)} neighbors, which is less than {face_classifier.k}"

    # Verify that the graph is symmetric
    symmetric_diff = set(map(tuple, edge_index.t().tolist())) ^ set(
        map(tuple, edge_index.flip(0).t().tolist())
    )
    assert len(symmetric_diff) == 0, "The k-nn graph is not symmetric"

    # Verify that there are no self-loops
    assert torch.all(
        edge_index[0] != edge_index[1]
    ), "The k-nn graph contains self-loops"
