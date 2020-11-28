import torch.nn
import bbml.nn


class Decoder(torch.nn.Module):

    def __init__(self, padding_mode=bbml.nn.PaddingMode.REFLECT, batch_momentum=0.1):
        super().__init__()

        model = [
            # Upsample blocks
            bbml.nn.UpsamplingBlock(in_channels=256, out_channels=128,
                                    batch_momentum=batch_momentum),

            bbml.nn.UpsamplingBlock(in_channels=128, out_channels=64,
                                    batch_momentum=batch_momentum),

            # Pad layer
            bbml.nn.Pad2DLayer(padding=3, padding_mode=padding_mode),

            # Output conv block
            bbml.nn.Conv2DBlock(in_channels=64, out_channels=3, kernel_size=7,
                                padding=0, batch_momentum=batch_momentum,
                                norm_type=bbml.nn.NormType.NONE,
                                activation_type=bbml.nn.ActivationType.TANH)
        ]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
