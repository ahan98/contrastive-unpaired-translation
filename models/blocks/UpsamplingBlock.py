import torch.nn as nn
from .ConvTranspose2DBlock import ConvTranspose2DBlock

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, up_factor=2, batch_momentum=0.1):
        super().__init__()

        self.model = ConvTranspose2DBlock(in_channels=in_channels, out_channels=out_channels,
                                          up_factor=up_factor, batch_momentum=batch_momentum)
        

    def forward(self, x):
        out = self.model(x)
        return out
