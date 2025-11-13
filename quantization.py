# === 0) Imports & Setup =======================================================
import torch
from torch import nn

# Dein Modell (ggf. anpassen)
from architectures_28x28.KKAN import KKAN_Small

# Neu: PT2E Kernfunktionen (keine Deprecated-Warnungen mehr)
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e
)

import torchvision.transforms as transforms

from tqdm import tqdm

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
  get_symmetric_quantization_config,
  XNNPACKQuantizer,
)

from torch.utils.data import DataLoader

from torchvision.datasets import FashionMNIST

import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    #model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def calibrate(model, data_loader, num_samples):
    #model.eval()
    seen = 0

    with torch.no_grad():
        pbar = tqdm(total=num_samples, desc="Calibrating", unit="sample")

        for images, targets in data_loader:
            batch_size = images.size(0)

            # Wenn wir über das Limit hinausgehen, nur einen Teil verarbeiten
            if seen + batch_size > num_samples:
                needed = num_samples - seen
                images = images[:needed]
                # targets = targets[:needed]  # falls du targets brauchst
                batch_size = needed

            model(images)
            seen += batch_size
            pbar.update(batch_size)

            if seen >= num_samples:
                break


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def quantize_kkan_small(model, dataset, num_samples: int = 100):
  test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

  example = (torch.randn(1, 1, 28, 28),)

  exported_model = torch.export.export(model, example).module()

  quantizer = XNNPACKQuantizer()
  quantizer.set_global(get_symmetric_quantization_config())

  prepared_model = prepare_pt2e(exported_model, quantizer)
  #print(prepared_model.graph)

  calibrate(prepared_model, test_loader, num_samples)

  quantized_model = convert_pt2e(prepared_model)
  #print(quantized_model)
  
  return quantized_model

criterion = nn.CrossEntropyLoss()

float_model = KKAN_Small(use_lut=False)
float_model.load_state_dict(torch.load("models/FashionMNIST_NoLUT/KKAN (Small) (gs = 5, NoLUT).pt", weights_only=False).state_dict()) 

model = KKAN_Small(use_lut=False)
model.load_state_dict(torch.load("models/FashionMNIST_NoLUT/KKAN (Small) (gs = 5, NoLUT).pt", weights_only=False).state_dict())


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_data = FashionMNIST(root='./data', train=True, download=True, transform=transform)


quantized_model = quantize_kkan_small(model, mnist_data)

mnist_data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
data_loader_test = DataLoader(mnist_data_test, batch_size=64, shuffle=False)

# Baseline model size and accuracy
print("Size of baseline model")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test)
print("Baseline Float Model Evaluation accuracy: %2.2f, %2.2f"%(top1.avg, top5.avg))

# Quantized model size and accuracy
print("Size of model after quantization")
# export again to remove unused weights
example_inputs = (torch.randn(1, 1, 28, 28),)
quantized_model = torch.export.export(quantized_model, example_inputs).module()
print_size_of_model(quantized_model)

data_loader_test = DataLoader(mnist_data_test, batch_size=1, shuffle=False)


top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

