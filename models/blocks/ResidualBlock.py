import torch.nn as nn
from .Conv2DBlock import Conv2DBlock
from .PadLayer import PadLayer
from .types import ActivationType, NormType, PaddingMode

''' (3x3 Convolution)-(BatchNorm)-(ReLU)-(3x3 Convolution)-(BatchNorm) '''


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=256, padding_mode=PaddingMode.REFLECT,
                 batch_momentum=0.1):

        super().__init__()

        model = [
            PadLayer(padding_mode=padding_mode),

            Conv2DBlock(in_channels=in_channels, out_channels=256,
                        batch_momentum=batch_momentum, norm_type=NormType.BATCH,
                        activation_type=ActivationType.RELU),

            PadLayer(padding_mode=padding_mode),

            Conv2DBlock(in_channels=256, out_channels=in_channels,
                        batch_momentum=batch_momentum, norm_type=NormType.BATCH,
                        activation_type=ActivationType.NONE)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
