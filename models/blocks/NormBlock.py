import torch.nn as nn
from .typing import NormalizationType

class NormBlock(nn.Module):
    def __init__(self, out_channels=256, batch_momentum=0.1, normalization_type=NormalizationType.INSTANCE):
        super().__init__()

        # Normalization layer
        if normalization_type == NormalizationType.BATCH:
            model = nn.BatchNorm2d(out_channels, momentum=batch_momentum)
        elif normalization_type == NormalizationType.INSTANCE:
            model = nn.InstanceNorm2d(out_channels, momentum=batch_momentum)

        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
