import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import Params


def get_transforms(params: Params, train: bool = True) -> transforms.Compose:
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


def get_loaders(params: Params) -> tuple[DataLoader, DataLoader]:
    train_tf = get_transforms(params, train=True)
    val_tf = get_transforms(params, train=False)

    if params.dataset == "mnist":
        train_ds = datasets.MNIST(params.data_dir, train=True, download=True, transform=train_tf)
        val_ds = datasets.MNIST(params.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params.data_dir, train=True, download=True, transform=train_tf)
        val_ds = datasets.CIFAR10(params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers
    )
    return train_loader, val_loader

def train_one_epoch(model: nn.Module,loader: DataLoader,optimizer: torch.optim.Optimizer,criterion: nn.Module,device: torch.device,log_interval: int) -> tuple[float, float]:
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


def run_training(model: nn.Module,params: Params,device: torch.device) -> None:
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay
    )

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

    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    print(f"Loss plot saved to: {params.plot_path}")