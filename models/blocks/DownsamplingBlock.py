import torch.nn as nn
from .Conv2DBlock import Conv2DBlock
from .types import NormType, PaddingMode

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128,
                 padding_mode=PaddingMode.REFLECT, batch_momentum=0.1):

        super().__init__()

        self.model = Conv2DBlock(in_channels=in_channels,
                                 out_channels=out_channels,
                                 padding_mode=padding_mode,
                                 batch_momentum=batch_momentum, stride=2,
                                 padding=1, norm_type=NormType.INSTANCE)

    def forward(self, x):
        out = self.model(x)
        return out
