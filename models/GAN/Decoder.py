import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from blocks.ResidualBlock import ResidualBlock
from blocks.UpsamplingBlock import UpsamplingBlock
from blocks.Conv2DBlock import Conv2DBlock
from blocks.typing_ import ActivationType, PaddingMode

class Decoder(nn.Module):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1,
                 padding_mode=PaddingMode.REFLECT):

        super().__init__()

        sequence = []

        # Residual blocks
        for _ in range(n_res_blocks):
            sequence += [ResidualBlock(padding_mode=padding_mode,
                                       batch_momentum=batch_momentum)]

        # Upsample blocks
        sequence += [UpsamplingBlock(in_channels=256, out_channels=128,
                                  batch_momentum=batch_momentum)]
        sequence += [UpsamplingBlock(in_channels=128, out_channels=64,
                                  batch_momentum=batch_momentum)]

        # Output conv block
        sequence += [Conv2DBlock(in_channels=64, out_channels=3, kernel_size=7,
                                 padding=3, batch_momentum=batch_momentum,
                                 activation_type=ActivationType.TANH,
                                 padding_mode=padding_mode)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out
