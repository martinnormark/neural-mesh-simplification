import pytest
import torch
from neural_mesh_simplification.models.layers.devconv import DevConv


@pytest.fixture
def sample_graph_data():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]], dtype=torch.long
    )
    return x, edge_index


def test_devconv_initialization():
    """Test if DevConv initializes correctly with expected input and output sizes."""
    conv = DevConv(in_channels=3, out_channels=64)
    assert isinstance(conv.W_theta, torch.nn.Linear)
    assert isinstance(conv.W_phi, torch.nn.Linear)
    assert conv.W_theta.in_features == 3
    assert conv.W_theta.out_features == 64
    assert conv.W_phi.in_features == 64  # Since W_phi is applied after scatter_max
    assert conv.W_phi.out_features == 64


def test_devconv_forward(sample_graph_data):
    """Test if DevConv produces correct output shapes."""
    x, edge_index = sample_graph_data
    conv = DevConv(in_channels=3, out_channels=64)

    output = conv(x, edge_index)

    assert output.shape == (x.shape[0], 64), "Output shape mismatch"


def test_devconv_no_nan_or_inf(sample_graph_data):
    """Ensure that DevConv does not produce NaN or Inf values."""
    x, edge_index = sample_graph_data
    conv = DevConv(in_channels=3, out_channels=64)

    output = conv(x, edge_index)

    assert not torch.isnan(output).any(), "NaN values detected in output"
    assert not torch.isinf(output).any(), "Inf values detected in output"


def test_devconv_gradient_flow(sample_graph_data):
    """Check if gradients properly flow through DevConv."""
    x, edge_index = sample_graph_data
    x.requires_grad_()
    conv = DevConv(in_channels=3, out_channels=64)

    output = conv(x, edge_index)
    loss = output.mean()  # Dummy loss
    loss.backward()

    assert x.grad is not None, "No gradient computed for input"
    assert x.grad.shape == x.shape, "Gradient shape mismatch"


def test_devconv_deterministic_output(sample_graph_data):
    """Ensure that DevConv produces deterministic results given the same input."""
    x, edge_index = sample_graph_data
    conv = DevConv(in_channels=3, out_channels=64)

    torch.manual_seed(42)
    out1 = conv(x, edge_index).detach()

    torch.manual_seed(42)
    out2 = conv(x, edge_index).detach()

    assert torch.allclose(out1, out2), "DevConv output is not deterministic"


def test_devconv_different_input_sizes():
    """Test DevConv with different input graph sizes to ensure flexibility."""
    conv = DevConv(in_channels=3, out_channels=64)

    x1 = torch.rand(10, 3)
    edge_index1 = torch.randint(0, 10, (2, 20))
    out1 = conv(x1, edge_index1)
    assert out1.shape == (10, 64)

    x2 = torch.rand(20, 3)
    edge_index2 = torch.randint(0, 20, (2, 40))
    out2 = conv(x2, edge_index2)
    assert out2.shape == (20, 64)
