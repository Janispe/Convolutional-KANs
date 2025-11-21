from torch import nn
import sys
import torch.nn.functional as F

# directory reach
# sys.path.append('../kan_convolutional')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0,0))
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=(0,0))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(125, 10)
        self.name = "CNN (Small)"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
    
class MediumCNN(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 28, num_classes: int = 10):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, padding=(0, 0))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            self._compute_flat_features(image_size),
            num_classes,
        )
        self.name = "CNN (Medium)"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def _conv_output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
        return (
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
            + 1
        )

    def _compute_flat_features(self, image_size: int) -> int:
        size = image_size
        size = self._conv_output_size(size, kernel_size=3, stride=1, padding=0)
        size = self._conv_output_size(size, kernel_size=2, stride=2, padding=0)
        size = self._conv_output_size(size, kernel_size=3, stride=1, padding=0)
        size = self._conv_output_size(size, kernel_size=2, stride=2, padding=0)
        return self.conv2.out_channels * size * size

class CNN_Big(nn.Module):
    def __init__(self):
        super(CNN_Big, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0,0))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, padding=(0,0))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(250, 100)
        self.fc2 = nn.Linear(100, 10)
        self.name = "CNN (Big)"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


    
class CNN_more_convs(nn.Module):
    def __init__(self):
        super(CNN_more_convs, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=(0, 0))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(500, 10)
        self.name = "CNN  (Medium, but with more convs)"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        # Definir una sola capa lineal con 'input_features' entradas y 10 salidas
        self.linear = nn.Linear(28*28, 10)
        self.flatten = nn.Flatten()
        self.name = "1 Layer MLP"


    def forward(self, x):
        # Pasar los datos a través de la capa lineal
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x
class CNN_deeper(nn.Module):
    def __init__(self):
        super(CNN_deeper, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(5, 25, kernel_size=3, padding=(0, 0))
        self.conv3 = nn.Conv2d(25, 25*5, kernel_size=2, padding=(0, 0))

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4*125, 10)
        self.name = "CNN 3 Conv layers & 1 MLP"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
