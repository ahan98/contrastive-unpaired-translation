import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from blocks.typing_ import PaddingMode, NormType, ActivationType
from blocks.Conv2DBlock import Conv2DBlock

class Discriminator(nn.Module):
    """
    From Section 7.2 of Zhu et al.:

    For discriminator networks, we use 70 × 70 PatchGAN. Let Ck denote a 4 × 4
    Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2. After
    the last layer, we apply a convolution to produce a 1-dimensional output. We
    do not use InstanceNorm for the ﬁrst C64 layer.  We use leaky ReLUs with a
    slope of 0.2. The discriminator architecture is: C64-C128-C256-C512
    """

    def __init__(self, batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):
        super().__init__()

        sequence = []
        filter_sizes = [64, 128, 256, 512]
        prev_out_channels = 3

        for n_filters in filter_sizes:

            # skip norm layer for first conv block
            norm_type = NormType.INSTANCE if n_filters != 64 else NormType.NONE
            stride = 2 if n_filters != 512 else 1

            sequence += [
                Conv2DBlock(in_channels=prev_out_channels, stride=stride,
                            out_channels=n_filters, kernel_size=4,
                            activation_type=ActivationType.LEAKY,
                            padding_mode=padding_mode, norm_type=norm_type,
                            batch_momentum=batch_momentum)
            ]

            prev_out_channels = n_filters

        # convolve result into one-dimensional output
        sequence += [
            Conv2DBlock(in_channels=prev_out_channels, out_channels=1,
                        kernel_size=4, padding_mode=padding_mode,
                        activation_type=ActivationType.NONE,
                        batch_momentum=batch_momentum)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out
