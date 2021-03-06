import torch.nn as nn
import torchvision.transforms as T

from .types import NormType, ActivationType
from .NormLayer import NormLayer
from .ActivationLayer import ActivationLayer


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=3, stride=1,
                 padding=1, batch_momentum=0.1, norm_type=NormType.INSTANCE,
                 activation_type=ActivationType.RELU):

        super().__init__()

        # BatchNorm uses learnable affine parameters, which includes its own
        # bias term, so only use bias for InstanceNorm.
        use_bias = (norm_type == NormType.INSTANCE)

        # Conv layer
        model = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding=padding, bias=use_bias),
        ]

        # Norm layer
        if norm_type != NormType.NONE:
            model += [NormLayer(norm_type, out_channels, batch_momentum)]

        # Activation layer
        if activation_type != ActivationType.NONE:
            model += [ActivationLayer(activation_type)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
