import torch.nn as nn
from .Conv2DBlock import Conv2DBlock
from .types import NormType


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128,
                 norm_type=NormType.INSTANCE, batch_momentum=0.1):

        super().__init__()

        self.model = Conv2DBlock(in_channels, out_channels, stride=2,
                                 batch_momentum=batch_momentum,
                                 norm_type=norm_type)

    def forward(self, x):
        out = self.model(x)
        return out
