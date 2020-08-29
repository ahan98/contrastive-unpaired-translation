import torch.nn as nn
from ..blocks.types import NormType, ActivationType
from ..blocks.Conv2DBlock import Conv2DBlock


class Discriminator(nn.Module):
    """
    From Section 7.2 of Zhu et al.:

    For discriminator networks, we use 70 × 70 PatchGAN. Let Ck denote a 4 × 4
    Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2. After
    the last layer, we apply a convolution to produce a 1-dimensional output. We
    do not use InstanceNorm for the ﬁrst C64 layer.  We use leaky ReLUs with a
    slope of 0.2. The discriminator architecture is: C64-C128-C256-C512
    """

    def __init__(self, batch_momentum=0.1):
        super().__init__()

        def patchGANConvLayer(in_channels, out_channels, stride, norm_type):
            return Conv2DBlock(in_channels, out_channels, stride=stride,
                               activation_type=ActivationType.LEAKY_RELU,
                               norm_type=norm_type, kernel_size=4)

        model = [
            patchGANConvLayer(3, 64, 2, NormType.NONE),
            patchGANConvLayer(64, 128, 2, NormType.INSTANCE),
            patchGANConvLayer(128, 256, 2, NormType.INSTANCE),
            patchGANConvLayer(256, 512, 1, NormType.INSTANCE),

            # convolve result into one-dimensional output
            Conv2DBlock(in_channels=512, out_channels=1, kernel_size=4,
                        norm_type=NormType.NONE,
                        activation_type=ActivationType.NONE)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
