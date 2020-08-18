import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from blocks.typing_ import ActivationType
from blocks.ActivationLayer import ActivationLayer

class MLP(nn.Module):
    """ 2-layer fully-connected network which outputs 256-dim vector """

    def __init__(self, in_channels=256, out_channels=256,
                 activation_type=ActivationType.RELU):
        super().__init__()
        sequence = [nn.Linear(in_channels, out_channels)]
        sequence += [ActivationLayer(activation_type)]
        sequence += [nn.Linear(out_channels, out_channels)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out
