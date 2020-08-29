import torch.nn as nn
from .types import ActivationType


class ActivationLayer(nn.Module):
    def __init__(self, activation_type):
        super().__init__()

        if activation_type == ActivationType.RELU:
            self.model = nn.ReLU()
        elif activation_type == ActivationType.TANH:
            self.model = nn.Tanh()
        elif activation_type == ActivationType.LEAKY_RELU:
            self.model = nn.LeakyReLU(0.2)
        elif activation_type == ActivationType.NONE:
            self.model = None

    def forward(self, x):
        if self.model is None:
            return x

        out = self.model(x)
        return out
