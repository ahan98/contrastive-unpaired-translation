import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from blocks.ResidualBlock import ResidualBlock
from blocks.UpsamplingBlock import UpsamplingBlock
from blocks.Conv2DBlock import Conv2DBlock
from blocks.types import ActivationType, PaddingMode

class Decoder(nn.Module):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):
        super().__init__()

        model = []

        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResidualBlock(padding_mode=padding_mode,
                                    batch_momentum=batch_momentum)]

        model += [
            # Upsample blocks
            UpsamplingBlock(in_channels=256, out_channels=128,
                            batch_momentum=batch_momentum),
            UpsamplingBlock(in_channels=128, out_channels=64,
                            batch_momentum=batch_momentum),

            # Output conv block
            Conv2DBlock(in_channels=64, out_channels=3, kernel_size=7,
                        padding=3, batch_momentum=batch_momentum,
                        activation_type=ActivationType.TANH,
                        padding_mode=padding_mode)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
