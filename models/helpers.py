import torch.nn as nn

def conv_block(in_channels=3, out_channels=256, kernel_size=3, stride=1,
              padding=1, padding_mode="reflect", batch_momentum=0.1,
              normtype="batch", relu=True):

    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                     padding_mode=padding_mode)
    if normtype == "batch":
        norm = nn.BatchNorm2d(out_channels, momentum=batch_momentum)
    elif normtype == "instance":
        norm = nn.InstanceNorm2d(out_channels, momentum=batch_momentum)

    modules = [conv, norm]
    if relu:
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)


