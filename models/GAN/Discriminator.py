import torch.nn as nn
from ..blocks.types import PaddingMode, NormType, ActivationType
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

    def __init__(self, batch_momentum=0.1, padding_mode=PaddingMode.REFLECT):
        super().__init__()

        def patchGANConvLayer(in_channels, out_channels, stride, norm_type):
            return Conv2DBlock(in_channels=in_channels, stride=stride,
                               out_channels=out_channels, kernel_size=4,
                               activation_type=ActivationType.LEAKY_RELU,
                               padding_mode=padding_mode, norm_type=norm_type,
                               batch_momentum=batch_momentum)

        model = [
            patchGANConvLayer(in_channels=3, out_channels=64, stride=2, norm_type=NormType.NONE),
            patchGANConvLayer(in_channels=64, out_channels=128, stride=2, norm_type=NormType.INSTANCE),
            patchGANConvLayer(in_channels=128, out_channels=256, stride=2, norm_type=NormType.INSTANCE),
            patchGANConvLayer(in_channels=256, out_channels=512, stride=1, norm_type=NormType.INSTANCE),

            # convolve result into one-dimensional output
            Conv2DBlock(in_channels=512, out_channels=1,
                        kernel_size=4, padding_mode=padding_mode,
                        activation_type=ActivationType.NONE,
                        batch_momentum=batch_momentum)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
