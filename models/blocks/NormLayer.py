import torch.nn as nn
from .types import NormType

class NormLayer(nn.Module):
    def __init__(self, norm_type, out_channels=256, batch_momentum=0.1):
        super().__init__()

        # Norm layer
        if norm_type == NormType.BATCH:
            self.model = nn.BatchNorm2d(out_channels, momentum=batch_momentum)
        elif norm_type == NormType.INSTANCE:
            self.model = nn.InstanceNorm2d(out_channels, momentum=batch_momentum)
        elif norm_type == NormType.NONE:
            self.model = None

    def forward(self, x):
        if self.model is None:
            return x

        out = self.model(x)
        return out
