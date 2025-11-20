import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from architectures_28x28.KKAN import KKAN_Small
from generic_train import simple_epoch_train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train = FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test = FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return train, test


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=128):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy}


def clone_with_layer_lut(trained_model, grid_size, spline_order, lut_size, device):
    """
    Create a copy of the trained weights inside a KKAN_Small variant
    whose layers run with the full layer-level LUT path.
    """
    layer_model = KKAN_Small(
        grid_size=grid_size,
        spline_order=spline_order,
        use_lut="layer",
        lut_size=lut_size,
    )
    load_result = layer_model.load_state_dict(
        trained_model.state_dict(), strict=False
    )
    missing, unexpected = load_result
    if unexpected:
        raise RuntimeError(
            "Unexpected parameters when cloning to layer-LUT model: "
            + ", ".join(unexpected)
        )
    if missing:
        print(
            "[LayerLUT] Skipping non-weight buffers while loading state dict: "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    layer_model.to(device)
    layer_model.eval()

    # Precompute lookup tables before inference for deterministic timing.
    with torch.no_grad():
        for module in layer_model.modules():
            rebuild_fn = getattr(module, "rebuild_layer_lut", None)
            if callable(rebuild_fn):
                try:
                    rebuild_fn(force=True)
                except RuntimeError:
                    # Some helper modules might not support forced rebuilds.
                    continue
    return layer_model


def main():
    train_ds, test_ds = build_datasets()
    config = {
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-3,
        "grid_size": 5,
        "spline_order": 3,
        "lut_size": 256,
        # Evaluate with progressively larger LUTs (defaults cover 2..256 in powers of two).
        "eval_lut_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        "train_use_lut": False,
    }

    baseline_model = KKAN_Small(
        grid_size=config["grid_size"],
        spline_order=config["spline_order"],
        use_lut=config["train_use_lut"],
        lut_size=config["lut_size"],
    )

    train_result = simple_epoch_train(
        baseline_model,
        train_ds=train_ds,
        device=device,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        test_ds=test_ds,
    )
    trained_model = train_result["model"]

    baseline_metrics = evaluate_model(
        trained_model, test_ds, device, batch_size=config["batch_size"]
    )
    print(
        "[Eval] Floating-point model "
        f"loss={baseline_metrics['loss']:.4f} "
        f"acc={baseline_metrics['accuracy']:.4f}"
    )

    eval_lut_sizes = config.get("eval_lut_sizes") or [config["lut_size"]]
    for lut_size in eval_lut_sizes:
        if lut_size < 2:
            print(f"[Eval] Skipping invalid lut_size={lut_size} (<2)")
            continue
        layer_lut_model = clone_with_layer_lut(
            trained_model=trained_model,
            grid_size=config["grid_size"],
            spline_order=config["spline_order"],
            lut_size=lut_size,
            device=device,
        )
        lut_metrics = evaluate_model(
            layer_lut_model, test_ds, device, batch_size=config["batch_size"]
        )
        print(
            "[Eval] Layer-LUT model "
            f"(lut_size={lut_size}) "
            f"loss={lut_metrics['loss']:.4f} "
            f"acc={lut_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
