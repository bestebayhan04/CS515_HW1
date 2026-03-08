import os
import torch
from torchviz import make_dot

from parameters import get_params
from main import build_model


def visualize_model() -> None:
    """
    Generate a computational graph visualization of the neural network.

    This function builds the model using the configuration parameters,
    performs a forward pass with a dummy input tensor, and uses the
    ``torchviz`` library to visualize the computational graph of the model.

    The visualization illustrates how data flows through the layers of the
    network and how the parameters are connected. The resulting graph is
    saved as a PNG image in the ``figures`` directory.

    The input tensor shape depends on the selected dataset:

    - MNIST:  (1, 1, 28, 28)
    - CIFAR-10: (1, 3, 32, 32)

    The generated file can be used in the report to illustrate the structure
    of the implemented neural network model.

    Output:
        figures/mlp_torchviz_graph.png
    """
    params = get_params()
    model = build_model(params)
    model.eval()

    if params.dataset == "mnist":
        x = torch.randn(1, 1, 28, 28)
    else:
        x = torch.randn(1, 3, 32, 32)

    y = model(x)

    os.makedirs("figures", exist_ok=True)

    dot = make_dot(
        y,
        params=dict(model.named_parameters())
    )

    dot.format = "png"
    dot.render("figures/mlp_torchviz_graph", cleanup=True)

    print("Model visualization saved to: figures/mlp_torchviz_graph.png")


if __name__ == "__main__":
    visualize_model()