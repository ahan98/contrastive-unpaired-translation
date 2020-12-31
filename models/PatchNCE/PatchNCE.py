import torch.nn
import torch.optim
import bbml.nn
import bbml.models.training as training

# Since this doesn't expose its model parameters, I'm not sure this is going to be affected by the optimizer
class PatchNCE(training.TrainableModel):

    def __init__(self, nc=256):
        self.l2norm = bbml.nn.NormalizeLayer(2)
        self.nc = nc
        super().__init__("patchNCE")

    # Hmmmm what is this??? I'm going to assume this works but it needs to be rewritten
    def create_mlp(self, samples):
        for mlp_id, feat in enumerate(samples):
            input_nc = feat.shape[1]
            mlp = torch.nn.Sequential(*[torch.nn.Linear(input_nc, self.nc),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.nc, self.nc)])
            mlp.cuda()
            for layer in mlp.children():
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.normal_(layer.weight.data, 0.0, 0.02)

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
