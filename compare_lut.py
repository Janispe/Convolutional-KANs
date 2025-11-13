from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import * 
from architectures_28x28.conv_and_kan import NormalConvsKAN,NormalConvsKAN_Medium
from generic_train import train_model_generic

from torchvision.datasets import FashionMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)


epochs = 5

model = KKAN_Small(use_lut=True)

train_model_generic(model, mnist_train, mnist_test,device,epochs = epochs, path="models/FashionMNIST_LUT")

model = KKAN_Small(use_lut=False)

train_model_generic(model, mnist_train, mnist_test,device,epochs = epochs, path="models/FashionMNIST_NoLUT")