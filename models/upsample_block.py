import torch.nn as nn
from .helpers import upsample_block

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, batch_momentum=0.1):
        super().__init__()
        self.block = upsample_block(in_channels, out_channels, batch_momentum)

    def forward(self, x):
        return self.block(x)
