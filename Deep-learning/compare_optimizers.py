import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_history(path: str) -> dict:
    """
    Load a training history file stored in JSON format.

    The JSON file is expected to contain metrics recorded during training,
    such as batch losses, training accuracy, and validation accuracy.

    Args:
        path (str):
            Path to the JSON history file.

    Returns:
        dict:
            Dictionary containing the stored training metrics.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    """
    Compare optimizer performance by visualizing training metrics.

    This script loads training history files produced during different
    optimizer experiments (Adam, SGD, SGD with Momentum, and RMSprop).
    It generates three plots:

    1. Training loss per iteration
    2. Training accuracy per epoch
    3. Validation accuracy per epoch

    The resulting visualization helps analyze how different optimizers
    affect convergence behavior and generalization performance.

    The figure is saved as ``optimizer_comparison.png``.
    """

    history_files = {
        "Adam": "optimizer_jsons/history_adam.json",
        "SGD": "optimizer_jsons/history_sgd.json",
        "SGDMomentum": "optimizer_jsons/history_sgdm.json",
        "RMSprop": "optimizer_jsons/history_rmsprop.json",
    }

    histories = {}
    for name, path in history_files.items():
        p = Path(path)
        if not p.exists():
            print(f"Missing file: {path}")
            return
        histories[name] = load_history(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    for name, h in histories.items():
        data_to_plot = h["batch_losses"][:600]
        axes[0].plot(
            data_to_plot,
            marker="o",
            linestyle="None",
            markersize=3,
            label=name
        )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlim(0, 600)
    axes[0].set_ylim(0.0, 2.5)
    axes[0].legend()

    for name, h in histories.items():
        epochs = range(len(h["train_accs"]))
        axes[1].plot(epochs, h["train_accs"], marker="o", label=name)
    axes[1].set_title("Training accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlim(0, 5)
    axes[1].legend()

    for name, h in histories.items():
        epochs = range(len(h["val_accs"]))
        axes[2].plot(epochs, h["val_accs"], marker="o", label=name)
    axes[2].set_title("Validation accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_xlim(0, 5)
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("optimizer_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()