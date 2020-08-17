import torch.nn as nn
from .Conv2DBlock import Conv2DBlock
from .typing import ActivationType

''' (3x3 Convolution)-(BatchNorm)-(ReLU)-(3x3 Convolution)-(BatchNorm) '''
class ResidualBlock(nn.Module):
    def __init__(self, batch_momentum=0.1):
        super().__init__()

        model = [
            Conv2DBlock(batch_momentum=batch_momentum, activation_type=ActivationType.RELU),
            Conv2DBlock(batch_momentum=batch_momentum, activation_type=ActivationType.NONE)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
