import torch.nn as nn
from .utils import NormalizationType, PaddingMode, padding_mode_to_str

class BlockBuilder:
    @staticmethod
    def norm_block(normtype: NormalizationType, out_channels: int, momentum: float):
        if normtype == NormalizationType.BATCH:
            return nn.BatchNorm2d(out_channels, momentum=batch_momentum)

        return nn.InstanceNorm2d(out_channels, momentum=batch_momentum)

    @staticmethod
    def conv_block(in_channels=3,
                   out_channels=256,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   padding_mode=PaddingMode.REFLECT,
                   batch_momentum=0.1,
                   normtype=NormalizationType.BATCH,
                   relu=True):

        modules = [
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              padding_mode=padding_mode_to_str(padding_mode)),

                    # block will switch on normtype
                    BlockBuilder.norm_block(normtype, out_channels, batch_momentum)
        ]

        if relu:
            modules += [nn.ReLU()]

        return nn.Sequential(*modules)

    @staticmethod
    def upsample_block(in_channels, out_channels, batch_momentum, up_factor=2):
        modules = [
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               stride=up_factor,
                               kernel_size=3,
                               padding=1,
                               output_padding=1),
            nn.InstanceNorm2d(out_channels,
                              momentum=batch_momentum),
            nn.ReLU()
        ]

        return nn.Sequential(*modules)