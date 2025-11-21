def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_lut_memory_bytes(model):
    """
    Return the total amount of memory (in bytes) currently used by LUT buffers.
    """
    total = 0
    for module in model.modules():
        lut_bytes_fn = getattr(module, "lut_memory_bytes", None)
        if callable(lut_bytes_fn):
            total += int(lut_bytes_fn())
    return total


def print_lut_memory_stats(model, prefix="[LUT]"):
    lut_bytes = get_model_lut_memory_bytes(model)
    if lut_bytes <= 0:
        print(f"{prefix} Model is not using LUT buffers.")
        return
    lut_mebibytes = lut_bytes / (1024 ** 2)
    print(f"{prefix} LUT buffers currently occupy {lut_bytes} bytes ({lut_mebibytes:.2f} MiB).")


try:
    from calflops import calculate_flops
except ImportError:
    calculate_flops = None


def infer_sample_input_shape(dataset):
    """
    Attempt to derive a 1-sample input shape tuple from the dataset.
    Returns None if the dataset is empty or not indexable.
    """
    try:
        if len(dataset) == 0:
            return None
    except TypeError:
        pass
    try:
        sample, _ = dataset[0]
    except Exception:
        return None
    if not isinstance(sample, (torch.Tensor,)):
        sample = torch.as_tensor(sample)
    return (1, *sample.shape)


def print_model_resource_stats(model, sample_input_shape=None):
    param_count = count_parameters(model)
    print(f"[Model] Trainable parameters: {param_count}")
    print_lut_memory_stats(model)
    if sample_input_shape is None:
        return
    if calculate_flops is None:
        print("[Model] calflops not available; skipping FLOP/MAC estimate.")
        return
    try:
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=sample_input_shape,
            output_as_string=False,
            output_precision=6,
            include_backPropagation=True,
            output_unit="G",
            print_results=False,
        )
        print(f"[Model] FLOPs: {flops}G | MACs: {macs}G | Params(reported): {params}")
        model.flops = flops
        model.macs = macs
    except Exception as exc:
        print(f"[Model] Failed to compute FLOPs via calflops: {exc}")


from evaluations import train_and_test_models
import torch.nn as nn
import torch.optim as optim
import time
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
def train_model_generic(model, train_ds, test_ds,device,epochs= 15,path =  "drive/MyDrive/KANs/models"):
    model.to(device)
    sample_input_shape = infer_sample_input_shape(train_ds)
    print_model_resource_stats(model, sample_input_shape)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    

    start = time.perf_counter()
    gen = torch.Generator()
    gen.manual_seed(0)
    mnist_train, mnist_val = torch.utils.data.random_split(train_ds, [51000,9000],
    generator=gen)
    # DataLoader
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, val_loader, optimizer, criterion, epochs=epochs, scheduler=scheduler,path= None)

    best_epochs = np.argmax(all_test_accuracy)+1
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=best_epochs, scheduler=scheduler,path= path,save_last=True)
    total_time = time.perf_counter() - start

    model.training_time = total_time/60 /epochs
    print("Total time (min)",total_time/60)
    if not path is None:
        saving_path = os.path.join(path,model.name+".pt")
        model =  torch.load(saving_path, map_location=torch.device(device), weights_only=False)
        model.train_losses = all_train_loss
        model.test_losses = all_test_loss
        torch.save(model,saving_path)
    print("Train loss",all_train_loss)
    print("Test loss",all_test_loss)

    #return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1


def simple_epoch_train(model, train_ds, device, epochs=5, batch_size=64, lr=1e-3, test_ds=None):
    model.to(device)
    sample_input_shape = infer_sample_input_shape(train_ds)
    print_model_resource_stats(model, sample_input_shape)
    model.train()
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    start_time = time.perf_counter()

    for epoch in range(epochs):
        cumulative_loss = 0.0
        progress_bar = tqdm(loader, desc=f"[SimpleTrain] Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
        seen_samples = 0
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item() * inputs.size(0)
            seen_samples += inputs.size(0)
            if seen_samples > 0:
                progress_bar.set_postfix(avg_loss=cumulative_loss / seen_samples)
        progress_bar.close()

        avg_loss = cumulative_loss / len(loader.dataset)
        train_losses.append(avg_loss)
        print(f"[SimpleTrain] Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    test_loss = None
    test_accuracy = None
    if test_ds is not None:
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        test_loss = total_loss / len(test_loader.dataset)
        test_accuracy = correct / total if total > 0 else 0.0
        model.train()

    training_time = time.perf_counter() - start_time

    return {
        "model": model,
        "train_losses": train_losses,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "parameter_count": count_parameters(model),
        "flops_G": getattr(model, "flops", None),
        "macs_G": getattr(model, "macs", None),
    }
