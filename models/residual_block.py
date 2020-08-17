import torch.nn as nn
import 

''' (3x3 Convolution)-(BatchNorm)-(ReLU)-(3x3 Convolution)-(BatchNorm) '''
class ResidualBlock(nn.Module):
    def __init__(self, batch_momentum=0.1):
        super().__init__()
        self.conv_bn_relu = conv_block(batch_momentum=batch_momentum)
        self.conv_bn = conv_block(in_channels=256,
                                  out_channels=3,
                                  batch_momentum=batch_momentum,
                                  relu=False)

    def forward(self, x):
        # in_channels = x.shape[1]
        residual = x
        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)
        out += residual
        return out

