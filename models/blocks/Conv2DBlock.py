import torch.nn as nn
from .typing import NormalizationType, ActivationType, PaddingMode, padding_mode_to_str
from .NormBlock import NormBlock

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=3,
                 stride=1, padding=1, padding_mode=PaddingMode.REFLECT, batch_momentum=0.1,
                 normalization_type=NormalizationType.BATCH, activation_type=ActivationType.RELU):
        super().__init__()

        padding_mode = padding_mode_to_str(padding_mode)

        model = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        ]

        # Normalization layer
        model += [NormBlock(out_channels, batch_momentum, normalization_type)]

        # Activation layer
        if activation_type == ActivationType.RELU:
            model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
