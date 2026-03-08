import os
import torch
from torchviz import make_dot

from parameters import get_params
from main import build_model


def visualize_model() -> None:
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