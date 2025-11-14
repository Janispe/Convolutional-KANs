import os
import torch
from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import * 
from architectures_28x28.conv_and_kan import NormalConvsKAN,NormalConvsKAN_Medium
from generic_train import simple_epoch_train

from torchvision.datasets import FashionMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)


epochs = 10
spline_order = 3


def train_and_save_model(use_lut: bool, save_dir: str, spline_order: int):
    model = KKAN_Small(use_lut=use_lut, spline_order=spline_order)
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


train_and_save_model(use_lut=True, save_dir="models/FashionMNIST_LUT", spline_order=spline_order)
train_and_save_model(use_lut=False, save_dir="models/FashionMNIST_NoLUT", spline_order=spline_order)
