import torch.nn as nn
from ..blocks.types import ActivationType
from ..blocks.ActivationLayer import ActivationLayer

class MLP(nn.Module):
    """ 2-layer fully-connected network which outputs 256-dim vector """

    def __init__(self, in_channels=256, out_channels=256,
                 activation_type=ActivationType.RELU):
        super().__init__()

        model = [
            nn.Linear(in_channels, out_channels),
            ActivationLayer(activation_type),
            nn.Linear(out_channels, out_channels)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
