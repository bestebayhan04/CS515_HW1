import torch

from parameters import Params
from train import get_loaders


@torch.no_grad()
def run_test(model: torch.nn.Module, params: Params, device: torch.device) -> float:
    """
    Evaluate a trained model on the test dataset.

    This function loads the saved model checkpoint, switches the model
    to evaluation mode, and computes classification accuracy on the test set.
    It also reports per-class accuracy for all classes.

    Args:
        model (torch.nn.Module):
            Neural network model to be evaluated.
        params (Params):
            Configuration object containing dataset settings,
            model parameters, and file paths.
        device (torch.device):
            Device used for computation (e.g., ``cpu`` or ``cuda``).

    Returns:
        float:
            Overall test accuracy.

    Notes:
        Gradients are disabled during evaluation using ``torch.no_grad()``
        for efficiency and reduced memory usage.
    """
    _, _, loader = get_loaders(params)

    model.load_state_dict(torch.load(params.save_path, map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct = [0] * params.num_classes
    class_total = [0] * params.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)

        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)

        for p, t in zip(preds, labels):
            t_idx = t.item()
            class_correct[t_idx] += (p == t).item()
            class_total[t_idx] += 1

    test_acc = correct / n
    print("\n=== Test Results ===")
    print(f"Test accuracy: {test_acc:.4f} ({correct}/{n})\n")

    for i in range(params.num_classes):
        acc = class_correct[i] / class_total[i]
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return test_acc