import torch.nn as nn
from ..blocks.PadLayer import PadLayer
from ..blocks.UpsamplingBlock import UpsamplingBlock
from ..blocks.Conv2DBlock import Conv2DBlock
from ..blocks.types import ActivationType, NormType, PaddingMode


class Decoder(nn.Module):

    def __init__(self, padding_mode=PaddingMode.REFLECT, batch_momentum=0.1):
        super().__init__()

        model = [
            # Upsample blocks
            UpsamplingBlock(in_channels=256, out_channels=128,
                            batch_momentum=batch_momentum),

            UpsamplingBlock(in_channels=128, out_channels=64,
                            batch_momentum=batch_momentum),

            # Pad layer
            PadLayer(padding=3, padding_mode=padding_mode),

            # Output conv block
            Conv2DBlock(in_channels=64, out_channels=3, kernel_size=7,
                        padding=0, batch_momentum=batch_momentum,
                        norm_type=NormType.NONE,
                        activation_type=ActivationType.TANH)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
