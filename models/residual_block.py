import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    # (3x3 Convolution)-(BatchNorm)-(ReLU)-(3x3 Convolution)-(BatchNorm)
    def __init__(self, in_channels, batch_momentum=0.1):
        super().__init__()
        self.conv_bn1 = conv_bn(in_channels, 256, batch_momentum)
        self.relu = nn.ReLU()
        self.conv_bn2 = conv_bn(256, in_channels, batch_momentum)

    def forward(self, x):
        in_channels = x.shape[1]
        residual = x
        out = self.conv_bn1(x)
        out = self.relu(out)
        out = self.conv_bn2(out)
        out += residual
        return out

def conv_bn(in_channels, out_channels, batch_momentum):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                  padding_mode="reflect"),
        nn.BatchNorm2d(num_features=out_channels, momentum=batch_momentum)
    )
