import argparse
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Params:
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