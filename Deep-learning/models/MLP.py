import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,input_size: int,hidden_sizes: list[int],num_classes: int,dropout: float = 0.3,activation: str = "relu",use_batchnorm: bool = False,) -> None:
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        layers.append(nn.Flatten())
        in_dim = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(activation_layer())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLP2(nn.Module):
    def __init__(self,input_dim: int = 784,hidden_dims: list[int] = [512, 256],num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)  # logits
        return x
    