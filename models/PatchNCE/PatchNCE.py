import torch.nn as nn
from .MLP import MLP
from ..blocks.Normalize import Normalize


class PatchNCE(nn.Module):

    def __init__(self):
        super().__init__()

        self.l2norm = Normalize(2)

    def forward(self, samples):
        features_final = {}

        def create_mlp(in_channels):
            return MLP(in_channels=in_channels).cuda()

        for layer_name, features in samples.items():
            in_channels = features.shape[-1]
            mlp = create_mlp(in_channels)

            features = mlp(features)
            features = self.l2norm(features)

            features_final[layer_name] = features

        return features_final
