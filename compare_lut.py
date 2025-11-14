import json
import os
import matplotlib.pyplot as plt
import torch
from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import *
from architectures_28x28.conv_and_kan import NormalConvsKAN, NormalConvsKAN_Medium
from generic_train import simple_epoch_train

from torchvision.datasets import FashionMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)


epochs = 10
spline_order = 3
lut_size = 4


def train_and_save_model(use_lut: bool, save_dir: str, spline_order: int, lut_size: int):
    model = KKAN_Small(use_lut=use_lut, spline_order=spline_order, lut_size=lut_size)
    result = simple_epoch_train(model, mnist_train, device, epochs=epochs, test_ds=mnist_test)
    trained_model = result["model"]
    trained_model.train_losses = result["train_losses"]
    trained_model.test_loss = result["test_loss"]
    trained_model.test_accuracy = result["test_accuracy"]
    trained_model.training_time = result["training_time_seconds"] / 60.0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{trained_model.name}.pt")
    torch.save(trained_model, save_path)
    print(f"[Save] Stored model at {save_path}")
    final_train_loss = trained_model.train_losses[-1] if trained_model.train_losses else None
    test_loss = trained_model.test_loss
    test_acc = trained_model.test_accuracy
    def fmt(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) and val is not None else "n/a"
    print(
        f"[Result] {trained_model.name} | "
        f"train_loss={fmt(final_train_loss)} "
        f"test_loss={fmt(test_loss)} "
        f"test_acc={fmt(test_acc)} "
        f"time_min={trained_model.training_time:.2f}"
    )
    summary = {
        "name": trained_model.name,
        "use_lut": use_lut,
        "spline_order": spline_order,
        "lut_size": lut_size,
        "train_loss": final_train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "training_time_min": trained_model.training_time,
        "save_path": save_path,
    }
    return summary


def plot_experiment_summary(results, output_path="results/compare_lut.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def filter_and_sort(predicate, key):
        filtered = [res for res in results if predicate(res) and res["test_accuracy"] is not None]
        return sorted(filtered, key=key)

    baseline = filter_and_sort(lambda r: r["lut_size"] == 4 and not r["use_lut"], key=lambda r: r["spline_order"])
    lut_baseline = filter_and_sort(lambda r: r["lut_size"] == 4 and r["use_lut"], key=lambda r: r["spline_order"])
    lut_sizes = filter_and_sort(lambda r: r["use_lut"] and r["spline_order"] == 3, key=lambda r: r["lut_size"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs spline order
    if baseline:
        axes[0].plot(
            [res["spline_order"] for res in baseline],
            [res["test_accuracy"] for res in baseline],
            marker="o",
            label="Ohne LUT",
        )
    if lut_baseline:
        axes[0].plot(
            [res["spline_order"] for res in lut_baseline],
            [res["test_accuracy"] for res in lut_baseline],
            marker="o",
            label="Mit LUT",
        )
    axes[0].set_title("Testgenauigkeit vs. Spline-Order (lut_size=4)")
    axes[0].set_xlabel("Spline-Order")
    axes[0].set_ylabel("Testgenauigkeit")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Accuracy vs LUT size for spline_order 3
    if lut_sizes:
        axes[1].plot(
            [res["lut_size"] for res in lut_sizes],
            [res["test_accuracy"] for res in lut_sizes],
            marker="o",
            color="#C44E52",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Testgenauigkeit vs. LUT-Größe (Spline-Order=3)")
    axes[1].set_xlabel("LUT-Größe")
    axes[1].set_ylabel("Testgenauigkeit")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("KKAN Fashion-MNIST Experimente")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"[Plot] Diagramm gespeichert unter {output_path}")


def save_results(results, output_path="results/compare_lut_metrics.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"[Results] Alle Metriken gespeichert in {output_path}")


experiment_results = []

for spline in range(1,5):
    experiment_results.append(
        train_and_save_model(use_lut=False, save_dir="models/FashionMNIST_NoLUT", spline_order=spline, lut_size=4)
    )

for spline in range(1,5):
    experiment_results.append(
        train_and_save_model(use_lut=True, save_dir="models/FashionMNIST_LUT", spline_order=spline, lut_size=4)
    )

for lut in [4, 8, 16, 32, 64]:
    experiment_results.append(
        train_and_save_model(use_lut=True, save_dir="models/FashionMNIST_LUT", spline_order=3, lut_size=lut)
    )

plot_experiment_summary(experiment_results)
save_results(experiment_results)
