import torch.nn as nn

def upsample_block(in_channels, out_channels, batch_momentum, up_factor=2):
    modules = []
    modules += [nn.ConvTranspose2d(in_channels, out_channels, stride=up_factor,
                                   kernel_size=3, padding=1, output_padding=1)]
    modules += [nn.InstanceNorm2d(out_channels, momentum=batch_momentum)]
    modules += [nn.ReLU()]

    return nn.Sequential(*modules)


def conv_block(in_channels=3, out_channels=256, kernel_size=3, stride=1,
              padding=1, padding_mode="reflect", batch_momentum=0.1,
              normtype="batch", relu=True):

    modules = []
    modules += [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, padding_mode=padding_mode)]
    if normtype == "batch":
        modules += [nn.BatchNorm2d(out_channels, momentum=batch_momentum)]
    elif normtype == "instance":
        modules += [nn.InstanceNorm2d(out_channels, momentum=batch_momentum)]

    if relu:
        modules += [nn.ReLU()]

    return nn.Sequential(*modules)
