import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from parameters import Params


def get_transforms(params: Params, train: bool = True) -> transforms.Compose:
    """
    Create the image transformation pipeline for the selected dataset.

    For MNIST, the transform consists of tensor conversion and normalization.
    For CIFAR-10, the training transform additionally includes data
    augmentation through random cropping and horizontal flipping.

    Args:
        params (Params):
            Configuration object containing dataset normalization values
            and dataset name.
        train (bool, optional):
            Whether to create the training transform. If ``True``, training
            augmentations are applied for CIFAR-10. Defaults to ``True``.

    Returns:
        transforms.Compose:
            Composed torchvision transform pipeline.
    """
    mean, std = params.mean, params.std

    if params.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(params: Params):
    """
    Create training, validation, and test data loaders.

    This function prepares dataset-specific transforms, downloads the
    selected dataset if needed, splits the original training set into
    training and validation subsets, and returns DataLoader objects
    for training, validation, and testing.

    Args:
        params (Params):
            Configuration object containing dataset name, data directory,
            batch size, number of workers, and random seed.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            Data loaders for the training, validation, and test sets.
    """

    train_tf = get_transforms(params, train=True)
    test_tf = get_transforms(params, train=False)

    if params.dataset == "mnist":
        full_train = datasets.MNIST(
            params.data_dir,
            train=True,
            download=True,
            transform=train_tf,
        )

        test_dataset = datasets.MNIST(
            params.data_dir,
            train=False,
            download=True,
            transform=test_tf,
        )

    else:  # cifar10
        full_train = datasets.CIFAR10(
            params.data_dir,
            train=True,
            download=True,
            transform=train_tf,
        )

        test_dataset = datasets.CIFAR10(
            params.data_dir,
            train=False,
            download=True,
            transform=test_tf,
        )

    # -------------------------------
    # split train -> train + val
    # -------------------------------

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(params.seed)
    )

    # -------------------------------
    # loaders
    # -------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )

    return train_loader, val_loader, test_loader

def train_one_epoch(model: nn.Module,loader: DataLoader,optimizer: torch.optim.Optimizer,criterion: nn.Module,device: torch.device,log_interval: int) -> tuple[float, float]:
    """
    Train the model for one epoch.

    This function performs one full pass over the training dataset,
    updating model parameters using backpropagation and the specified
    optimizer. It also computes the average loss and accuracy.

    Args:
        model (nn.Module):
            Neural network model to train.
        loader (DataLoader):
            Data loader for the training set.
        optimizer (torch.optim.Optimizer):
            Optimizer used to update model parameters.
        criterion (nn.Module):
            Loss function used for optimization.
        device (torch.device):
            Device used for computation.
        log_interval (int):
            Number of batches between logging updates.

    Returns:
        tuple[float, float]:
            Average training loss and training accuracy for the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx + 1}/{len(loader)}] "
                  f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")

    return total_loss / n, correct / n


def validate(model: nn.Module,loader: DataLoader,criterion: nn.Module,device: torch.device) -> tuple[float, float]:
    """
    Evaluate the model on a validation dataset.

    This function runs the model in evaluation mode without gradient
    computation and returns the average loss and accuracy on the
    provided dataset.

    Args:
        model (nn.Module):
            Neural network model to evaluate.
        loader (DataLoader):
            Data loader for the validation set.
        criterion (nn.Module):
            Loss function used for evaluation.
        device (torch.device):
            Device used for computation.

    Returns:
        tuple[float, float]:
            Average validation loss and validation accuracy.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            total_loss += loss.detach().item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)

    return total_loss / n, correct / n


def save_loss_plot(train_losses: list[float],val_losses: list[float],plot_path: str) -> None:
    """
    Save a plot of training and validation loss curves.

    Args:
        train_losses (list[float]):
            List of average training losses for each epoch.
        val_losses (list[float]):
            List of average validation losses for each epoch.
        plot_path (str):
            File path where the plot image will be saved.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def build_optimizer(model: nn.Module, params: Params) -> torch.optim.Optimizer:
    """
    Build and return the optimizer specified in the configuration.

    Supported optimizers include Adam, SGD, SGD with momentum,
    and RMSprop. The optimizer hyperparameters are taken from
    the provided Params object.

    Args:
        model (nn.Module):
            Neural network model whose parameters will be optimized.
        params (Params):
            Configuration object containing optimizer type and
            related hyperparameters such as learning rate,
            weight decay, momentum, and RMSprop alpha.

    Returns:
        torch.optim.Optimizer:
            Initialized optimizer for the model parameters.

    Raises:
        ValueError:
            If the optimizer name is not supported.
    """
    if params.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay
        )

    if params.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay
        )

    if params.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=params.learning_rate,
            momentum=params.momentum,
            weight_decay=params.weight_decay
        )

    if params.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=params.learning_rate,
            alpha=params.rmsprop_alpha,
            weight_decay=params.weight_decay
        )

    raise ValueError(f"Unsupported optimizer: {params.optimizer}")

def run_training(model: nn.Module,params: Params,device: torch.device) -> None:
    """
    Run the full training pipeline.

    This function prepares data loaders, loss function, optimizer, and
    learning-rate scheduler; trains the model for multiple epochs;
    tracks training and validation metrics; applies early stopping;
    saves the best model checkpoint; and stores the loss curve plot.

    Args:
        model (nn.Module):
            Neural network model to train.
        params (Params):
            Configuration object containing dataset, training, and
            optimization parameters.
        device (torch.device):
            Device used for computation.
    """
    train_loader, val_loader, test_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, params)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.scheduler_step_size,
        gamma=params.scheduler_gamma
    )

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, params.log_interval
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params.save_path)
            patience_counter = 0
            print(f"  Saved best model (val_acc={best_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  Early stop counter: {patience_counter}/{params.early_stop}")

        if patience_counter >= params.early_stop:
            print("  Early stopping triggered.")
            break

    model.load_state_dict(best_weights)
    save_loss_plot(train_losses, val_losses, params.plot_path)
    print(f"Loss plot saved to: {params.plot_path}")

    print(f"\nFinal train accuracy: {train_accs[-1]:.4f}")
    print(f"Best validation accuracy: {best_acc:.4f}")