import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for image classification.

    This model first flattens the input tensor and then applies a sequence of
    fully connected layers. Each hidden layer may optionally include batch
    normalization, a non-linear activation, and dropout. The final layer
    produces class logits.

    Args:
        input_size (int):
            Number of input features after flattening.
        hidden_sizes (list[int]):
            Sizes of the hidden fully connected layers.
        num_classes (int):
            Number of output classes.
        dropout (float, optional):
            Dropout probability applied after each hidden activation.
            Defaults to 0.3.
        activation (str, optional):
            Activation function used in hidden layers. Supported values are
            ``"relu"`` and ``"gelu"``. Defaults to ``"relu"``.
        use_batchnorm (bool, optional):
            Whether to apply batch normalization before activation in each
            hidden layer. Defaults to ``False``.

    Attributes:
        net (nn.Sequential):
            Sequential container holding the full MLP architecture.

    Shape:
        Input:
            ``(N, *)`` where ``N`` is the batch size.
        Output:
            ``(N, num_classes)`` containing class logits.
    """
    def __init__(self,input_size: int,hidden_sizes: list[int],num_classes: int,dropout: float = 0.3,activation: str = "relu",use_batchnorm: bool = False,) -> None:
        """
        Initialize the MLP model.

        Args:
            input_size (int):
                Number of flattened input features.
            hidden_sizes (list[int]):
                List of hidden layer dimensions.
            num_classes (int):
                Number of output classes.
            dropout (float, optional):
                Dropout probability. Defaults to 0.3.
            activation (str, optional):
                Hidden-layer activation type. Must be ``"relu"`` or ``"gelu"``.
                Defaults to ``"relu"``.
            use_batchnorm (bool, optional):
                If ``True``, applies batch normalization after each linear layer
                and before the activation. Defaults to ``False``.

        Raises:
            ValueError:
                If an unsupported activation function is provided.
        """
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
        """
        Run the forward pass of the MLP.

        Args:
            x (torch.Tensor):
                Input tensor of images or flattened features.

        Returns:
            torch.Tensor:
                Output logits for each class.
        """
        return self.net(x)

class MLP2(nn.Module):
    """
    Alternative MLP implementation using ``nn.ModuleList``.

    This version explicitly stores the hidden linear layers inside a
    ``ModuleList`` and applies ReLU activation after each hidden layer during
    the forward pass. The input is first flattened, then passed through all
    hidden layers, and finally mapped to class logits.

    Args:
        input_dim (int, optional):
            Number of input features after flattening. Defaults to 784.
        hidden_dims (list[int], optional):
            Sizes of hidden layers. Defaults to ``[512, 256]``.
        num_classes (int, optional):
            Number of output classes. Defaults to 10.

    Attributes:
        flatten (nn.Flatten):
            Layer used to flatten the input tensor.
        hidden_layers (nn.ModuleList):
            List of hidden fully connected layers.
        output_layer (nn.Linear):
            Final linear layer producing class logits.

    Shape:
        Input:
            ``(N, *)`` where ``N`` is the batch size.
        Output:
            ``(N, num_classes)`` containing class logits.
    """
    def __init__(self,input_dim: int = 784,hidden_dims: list[int] = [512, 256],num_classes: int = 10) -> None:
        """
        Initialize the alternative MLP model.

        Args:
            input_dim (int, optional):
                Number of flattened input features. Defaults to 784.
            hidden_dims (list[int], optional):
                List of hidden layer sizes. Defaults to ``[512, 256]``.
            num_classes (int, optional):
                Number of output classes. Defaults to 10.
        """
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
        """
        Run the forward pass of the alternative MLP.

        Args:
            x (torch.Tensor):
                Input tensor of images or flattened features.

        Returns:
            torch.Tensor:
                Output logits for each class.
        """
        x = self.flatten(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)  # logits
        return x
    