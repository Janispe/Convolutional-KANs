from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')

from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANLinear import KANLinear

_DEFAULT_ACTIVATION = object()

class KKAN_Small(nn.Module):
    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        use_lut: bool = False,
        lut_size: int = 4,
        activation: nn.Module | None | bool = _DEFAULT_ACTIVATION,
        in_channels: int = 1,
        image_size: int = 28,
    ):
        super().__init__()
        conv_channels = 5
        self.conv1 = KAN_Convolutional_Layer(in_channels=in_channels,
            out_channels= conv_channels,
            kernel_size= (3,3),
            grid_size = grid_size,
            spline_order=spline_order,
            padding =(0,0),
            use_lut=use_lut,
            lut_size=lut_size,
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=conv_channels,
            out_channels= conv_channels,
            kernel_size = (3,3),
            grid_size = grid_size,
            spline_order=spline_order,
            padding =(0,0),
            use_lut=use_lut,
            lut_size=lut_size,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )

        # Allow caller to pick activation; keep tanh as default. Pass False or None to disable.
        if activation is _DEFAULT_ACTIVATION:
            self.activation = nn.Tanh()
        elif activation is False or activation is None:
            self.activation = None
        else:
            self.activation = activation
        
        self.flat = nn.Flatten() 

        self.kan1 = KANLinear(
            self._compute_flat_features(
                image_size=image_size,
                conv_channels=conv_channels,
            ),
            10,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
            use_lut=use_lut,
            lut_size=lut_size,
        )
        lut_flag = (
            f"LUT (ls = {lut_size})"
            if use_lut
            else "NoLUT"
        )
        self.name = f"KKAN (Small) (gs = {grid_size}, so = {spline_order}, {lut_flag})"


    def forward(self, x):
        if self.activation is not None:
            x = self.activation(self.conv1(x))
        else:
            x = self.conv1(x)

        x = self.pool1(x)

        if self.activation is not None:
            x = self.activation(self.conv2(x))
        else:
            x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)

        return x

    @staticmethod
    def _conv_output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
        return (
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
            + 1
        )

    def _compute_flat_features(self, image_size: int, conv_channels: int) -> int:
        size = image_size
        size = self._conv_output_size(size, kernel_size=3, stride=1, padding=0)
        size = self._conv_output_size(size, kernel_size=2, stride=2, padding=0)
        size = self._conv_output_size(size, kernel_size=3, stride=1, padding=0)
        size = self._conv_output_size(size, kernel_size=2, stride=2, padding=0)
        return conv_channels * size * size

class KKAN_Convolutional_Network(nn.Module):
    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=1,
            out_channels= 5,
            kernel_size= (3,3),
            grid_size = grid_size,
            padding =(0,0)
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=5,
            out_channels= 10,
            kernel_size = (3,3),
            grid_size = grid_size,
            padding =(0,0)

        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 

        self.kan1 = KANLinear(
            250,
            10,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )
        self.name = f"KKAN (Medium) (gs = {grid_size})"


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x) 
        x = F.log_softmax(x, dim=1)

        return x
    
