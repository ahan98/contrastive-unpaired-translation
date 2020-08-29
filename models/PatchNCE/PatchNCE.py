import torch.nn as nn
from .MLP import MLP


class PatchNCE(nn.Module):

    def __init__(self, encoder, image_height=256, image_width=256):
        super().__init__()

        self.encoder = encoder

    def forward(self, x):
        _, samples = self.encoder(x)
        features_final = {}

        def create_mlp(in_channels):
            return MLP(in_channels=in_channels).cuda()

        for layer_name, features in samples.items():
            in_channels = features.shape[-1]
            mlp = create_mlp(in_channels)
            features = mlp(features)

            L2_norm = features.norm(p=2, dim=1, keepdim=True)  # L2 norm
            features_norm = features.div(L2_norm + 1e-7)

            features_final[layer_name] = features_norm

        return features_final
