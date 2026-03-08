import random
import ssl
import numpy as np
import torch

from parameters import get_params, Params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from train import run_training
from test import run_test


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    This function initializes random seeds for Python, NumPy, and PyTorch
    to ensure deterministic behavior during training and evaluation.

    Args:
        seed (int):
            Seed value used for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params: Params) -> torch.nn.Module:
    """
    Construct a neural network model based on the provided configuration.

    The function selects the appropriate architecture depending on the
    model name and dataset specified in the ``Params`` object.

    Args:
        params (Params):
            Configuration object containing model and dataset parameters.

    Returns:
        torch.nn.Module:
            Instantiated neural network model.

    Raises:
        ValueError:
            If an unsupported model name is provided or if an incompatible
            model–dataset combination is selected.
    """
    model_name = params.model
    dataset = params.dataset
    nc = params.num_classes

    if model_name == "mlp":
        return MLP(
            input_size=params.input_size,
            hidden_sizes=params.hidden_sizes,
            num_classes=nc,
            dropout=params.dropout,
            activation=params.activation,
            use_batchnorm=params.use_batchnorm,
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=params.vgg_depth, num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, params.resnet_layers, num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    """
    Main entry point for running the training and evaluation pipeline.

    This function parses command-line parameters, initializes the random
    seed, selects the computation device, builds the requested model, and
    runs training and/or testing depending on the chosen execution mode.

    Execution modes:
        - ``train``: train the model only
        - ``test``: evaluate a saved model
        - ``both``: train the model and then evaluate it
    """
    params = get_params()

    set_seed(params.seed)
    print(f"Seed set to: {params.seed}")
    print(f"Dataset: {params.dataset}  |  Model: {params.model}")

    device = torch.device(
        params.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params.mode in ("train", "both"):
        run_training(model, params, device)

    if params.mode in ("test", "both"):
        test_acc = run_test(model, params, device)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()