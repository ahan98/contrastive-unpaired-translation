import torch.nn as nn
from .types import ActivationType

class ActivationLayer(nn.Module):
    def __init__(self, activation_type):
        super().__init__()

        if activation_type == ActivationType.RELU:
            model = nn.ReLU()
        elif activation_type == ActivationType.TANH:
            model = nn.Tanh()
        elif activation_type == ActivationType.LEAKY_RELU:
            model = nn.LeakyReLU(0.2)
        elif activation_type == ActivationType.NONE:
            model = None

        self.model = model

    def forward(self, x):
        if self.model is None:
            return x

        out = self.model(x)
        return out
