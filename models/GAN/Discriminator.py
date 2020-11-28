from abc import ABC
import torch.optim
import bbml.models.training as training
import bbml.nn


class Discriminator(training.TrainableModel, ABC):
    """
    From Section 7.2 of Zhu et al.:

    For discriminator networks, we use 70 × 70 PatchGAN. Let Ck denote a 4 × 4
    Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2. After
    the last layer, we apply a convolution to produce a 1-dimensional output. We
    do not use InstanceNorm for the ﬁrst C64 layer.  We use leaky ReLUs with a
    slope of 0.2. The discriminator architecture is: C64-C128-C256-C512
    """

    def __init__(self, batch_momentum=0.1):
        def patch_gan_conv_layer(in_channels, out_channels, stride, norm_type):
            return bbml.nn.Conv2DBlock(in_channels, out_channels, stride=stride,
                                       activation_type=bbml.nn.ActivationType.LEAKY_RELU,
                                       norm_type=norm_type, kernel_size=4, batch_momentum=batch_momentum)

        model = [
            patch_gan_conv_layer(3, 64, 2, bbml.nn.NormType.NONE),
            patch_gan_conv_layer(64, 128, 2, bbml.nn.NormType.INSTANCE),
            patch_gan_conv_layer(128, 256, 2, bbml.nn.NormType.INSTANCE),
            patch_gan_conv_layer(256, 512, 1, bbml.nn.NormType.INSTANCE),

            # convolve result into one-dimensional output
            bbml.nn.Conv2DBlock(in_channels=512, out_channels=1, kernel_size=4,
                                norm_type=bbml.nn.NormType.NONE,
                                activation_type=bbml.nn.ActivationType.NONE,
                                batch_momentum=batch_momentum)
        ]

        super().__init__(torch.nn.Sequential(*model))
