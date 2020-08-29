import torch.nn as nn
from .types import NormType, ActivationType
from .NormLayer import NormLayer
from .ActivationLayer import ActivationLayer


class ConvTranspose2DBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=3,
                 up_factor=2, padding=1, batch_momentum=0.1,
                 activation_type=ActivationType.RELU):

        super().__init__()

        stride = up_factor

        model = [
            # Conv transpose layer
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding, output_padding=1),

            # Norm layer
            NormLayer(NormType.INSTANCE, out_channels, batch_momentum)
        ]

        # Activation layer
        if activation_type != ActivationType.NONE:
            model += [ActivationLayer(activation_type)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
