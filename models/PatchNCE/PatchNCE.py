import torch.nn as nn
from .MLP import MLP
from ..blocks.Normalize import Normalize


class PatchNCE(nn.Module):

    def __init__(self, norm_pow=2):
        super().__init__()

        self.l2norm = Normalize(norm_pow)
        self.create_mlp = MLP

    def forward(self, samples):
        features_final = {}
        for layer_name, features in samples.items():
            in_channels = features.shape[-1]
            mlp = self.create_mlp(in_channels).cuda()

            features = mlp(features)
            features = self.l2norm(features)

            features_final[layer_name] = features

        return features_final
