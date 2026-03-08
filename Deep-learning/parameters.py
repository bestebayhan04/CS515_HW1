import argparse
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Params:
    """
    Container for experiment and runtime configuration parameters.

    This dataclass stores dataset settings, model architecture parameters,
    training hyperparameters, miscellaneous runtime options, and command-line
    execution mode. It is used to pass configuration values throughout the
    project in a structured way.

    Attributes:
        dataset (str):
            Dataset name. Supported values are ``"mnist"`` and ``"cifar10"``.
        data_dir (str):
            Directory where dataset files are stored or downloaded.
        num_workers (int):
            Number of worker processes used by data loaders.
        mean (Tuple[float, ...]):
            Channel-wise dataset mean used for normalization.
        std (Tuple[float, ...]):
            Channel-wise dataset standard deviation used for normalization.
        model (str):
            Model type. Supported values include ``"mlp"``, ``"cnn"``,
            ``"vgg"``, and ``"resnet"``.
        input_size (int):
            Flattened input size for the selected dataset.
        hidden_sizes (List[int]):
            Hidden layer sizes for the MLP model.
        num_classes (int):
            Number of output classes.
        dropout (float):
            Dropout probability used in the MLP model.
        activation (str):
            Activation function name for the MLP model.
        use_batchnorm (bool):
            Whether batch normalization is enabled in MLP hidden layers.
        vgg_depth (str):
            Depth configuration for the VGG model.
        resnet_layers (List[int]):
            Number of blocks in each ResNet stage.
        epochs (int):
            Number of training epochs.
        batch_size (int):
            Batch size used during training and evaluation.
        learning_rate (float):
            Optimizer learning rate.
        weight_decay (float):
            L2 regularization coefficient.
        scheduler_step_size (int):
            Step interval for the learning rate scheduler.
        scheduler_gamma (float):
            Multiplicative decay factor for the scheduler.
        early_stop (int):
            Early stopping patience in epochs.
        optimizer (str):
            Optimizer type.
        momentum (float):
            Momentum value used by SGD with momentum.
        rmsprop_alpha (float):
            Alpha parameter used by RMSprop.
        seed (int):
            Random seed for reproducibility.
        device (str):
            Device used for computation, such as ``"cpu"`` or ``"cuda"``.
        save_path (str):
            File path where the best model checkpoint is saved.
        plot_path (str):
            File path used to save the loss plot.
        log_interval (int):
            Logging frequency during training.
        mode (str):
            Execution mode. Supported values are ``"train"``, ``"test"``,
            and ``"both"``.
    """

    # Data
    dataset: str
    data_dir: str
    num_workers: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    # Model
    model: str
    input_size: int
    hidden_sizes: List[int]
    num_classes: int
    dropout: float
    activation: str
    use_batchnorm: bool
    vgg_depth: str
    resnet_layers: List[int]

    # Training
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    scheduler_step_size: int
    scheduler_gamma: float
    early_stop: int
    optimizer: str
    momentum: float
    rmsprop_alpha: float

    # Misc
    seed: int
    device: str
    save_path: str
    plot_path: str
    log_interval: int

    # CLI
    mode: str


def get_params() -> Params:
    """
    Parse command-line arguments and construct a ``Params`` object.

    This function defines the command-line interface for the project,
    including dataset selection, model configuration, and training
    hyperparameters. It also assigns dataset-specific normalization values
    and input dimensions before returning all settings as a structured
    ``Params`` instance.

    Returns:
        Params:
            A populated configuration object containing all runtime,
            model, dataset, and training parameters.
    """
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet"], default="mlp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)

    # MLP-specific additions
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer sizes for MLP"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout ratio for MLP"
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu"],
        default="relu",
        help="Activation function for MLP"
    )
    parser.add_argument(
        "--use_batchnorm",
        action="store_true",
        help="Enable BatchNorm1d in MLP hidden layers"
    )

    # Training-framework additions
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient"
    )
    parser.add_argument(
        "--scheduler_step_size",
        type=int,
        default=5,
        help="Step size for StepLR"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.5,
        help="Gamma for StepLR"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd", "sgd_momentum", "rmsprop"],
        default="adam",
        help="Optimizer type"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD with momentum"
    )
    parser.add_argument(
        "--rmsprop_alpha",
        type=float,
        default=0.99,
        help="Alpha parameter for RMSprop"
    )

    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")

    # ResNet-specific
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)"
    )

    args = parser.parse_args()

    if args.dataset == "mnist":
        input_size = 784
        mean, std = (0.1307,), (0.3081,)
    else:
        input_size = 3072
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    return Params(
        # Data
        dataset=args.dataset,
        data_dir="./data",
        num_workers=2,
        mean=mean,
        std=std,

        # Model
        model=args.model,
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=10,
        dropout=args.dropout,
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,

        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        early_stop=args.early_stop,
        optimizer=args.optimizer,
        momentum=args.momentum,
        rmsprop_alpha=args.rmsprop_alpha,

        # Misc
        seed=42,
        device=args.device,
        save_path="best_model.pth",
        plot_path="loss_curve.png",
        log_interval=100,

        # CLI
        mode=args.mode,
    )