import torch.nn as nn
from .typing_ import ActivationType

class ActivationLayer(nn.Module):
    def __init__(self, activation_type):
        super().__init__()

        if activation_type == ActivationType.RELU:
            model = nn.ReLU()
        elif activation_type == ActivationType.TANH:
            model = nn.Tanh()
        elif activation_type == ActivationType.LEAKY:
            model = nn.LeakyReLU(0.2)

        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out
