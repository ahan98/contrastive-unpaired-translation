import torch.nn as nn
from .typing_ import NormType

class NormLayer(nn.Module):
    def __init__(self, norm_type, out_channels=256, batch_momentum=0.1):
        super().__init__()

        # Norm layer
        if norm_type == NormType.BATCH:
            model = nn.BatchNorm2d(out_channels, momentum=batch_momentum)
        elif norm_type == NormType.INSTANCE:
            model = nn.InstanceNorm2d(out_channels, momentum=batch_momentum)

        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
