import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch.nn as nn
from GAN.Encoder import Encoder
from .MLP import MLP

class PatchNCE(nn.Module):

    def __init__(self, encoder, image_height=256, image_width=256):
        super().__init__()

        self.encoder = encoder
        self.MLP = MLP()

    def forward(self, x):
        out, samples = self.encoder(x)
        features_final = {}
        for layer_key in samples:
            layer_sample = samples[layer_key]
            features_final[layer_key] = self.MLP(layer_sample)
        return features_final

