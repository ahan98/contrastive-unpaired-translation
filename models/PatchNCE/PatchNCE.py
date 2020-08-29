import torch.nn as nn
from ..blocks.Normalize import Normalize


class PatchNCE(nn.Module):

    def __init__(self, nc=256):
        super().__init__()

        self.l2norm = Normalize(2)
        self.nc = nc

    def create_mlp(self, samples):
        for mlp_id, feat in enumerate(samples):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(),
                                  nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            for layer in mlp.children():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight.data, 0.0, 0.02)

            setattr(self, 'mlp_%d' % mlp_id, mlp)

    def forward(self, samples):
        features_final = []
        self.create_mlp(samples)

        for mlp_id, features in enumerate(samples):
            mlp = getattr(self, 'mlp_%d' % mlp_id)
            features = mlp(features)
            features = self.l2norm(features)

            features_final.append(features)

        return features_final
