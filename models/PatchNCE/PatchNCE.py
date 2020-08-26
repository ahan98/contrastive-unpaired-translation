import torch.nn as nn
from .MLP import MLP

class PatchNCE(nn.Module):

    def __init__(self, encoder, image_height=256, image_width=256):
        super().__init__()

        self.encoder = encoder
        self.MLP = MLP()

    def forward(self, x):
        _, samples = self.encoder(x)
        features_final = {}
        for layer_key in samples:
            features = self.MLP(samples[layer_key])
            norm = features.norm(p=2, dim=1, keepdim=True)  # L2 norm
            features_norm = features.div(norm)
            features_final[layer_key] = features_norm

        return features_final
