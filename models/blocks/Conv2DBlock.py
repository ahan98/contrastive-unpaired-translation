import torch.nn as nn
from .typing import NormalizationType, ActivationType, PaddingMode, padding_mode_to_str

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=3, 
                 stride=1, padding=1, padding_mode=PaddingMode.REFLECT, batch_momentum=0.1,
                 normilzation_type=NormalizationType.BATCH, activation_type=ActivationType.RELU):
        super().__init__()

        padding_mode = padding_mode_to_str(padding_mode)

        model = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        ]

        # TODO: we can clean this up more by making ActivationBlocks and NormBlocks which take a type + other args
        # then we don't have to do this weird switch thing everywhere that takes norm and activation types as an argument

        # Normalization layer
        if normilzation_type == NormalizationType.BATCH:
            model += [nn.BatchNorm2d(out_channels, momentum=batch_momentum)]
        elif normilzation_type == NormalizationType.INSTANCE:
            model += [nn.InstanceNorm2d(out_channels, momentum=batch_momentum)]

        # Activation layer
        if activation_type == ActivationType.RELU:
            model += [nn.ReLU()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
