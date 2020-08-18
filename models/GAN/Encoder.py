import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from blocks.Conv2DBlock import Conv2DBlock
from blocks.DownsamplingBlock import DownsamplingBlock
from blocks.ResidualBlock import ResidualBlock
from blocks.typing_ import PaddingMode

class Encoder(nn.Module):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1,
                 padding_mode=PaddingMode.REFLECT, sample_size=256):
        super().__init__()

        self.sample_size = sample_size

        self.conv = Conv2DBlock(out_channels=64, kernel_size=7, padding=3,
                                batch_momentum=batch_momentum,
                                padding_mode=padding_mode)

        self.dsample1 = DownsamplingBlock(in_channels=64, out_channels=128,
                                          batch_momentum=batch_momentum,
                                          padding_mode=padding_mode)

        self.dsample2 = DownsamplingBlock(in_channels=128, out_channels=256,
                                          batch_momentum=batch_momentum,
                                          padding_mode=padding_mode)

        self.res_blocks = []
        for _ in range(n_res_blocks):
            self.res_blocks += [ResidualBlock(padding_mode=padding_mode,
                                              batch_momentum=batch_momentum)]

    def forward(self, x):
        """
        From Section C.1 of Park et al.:

        In order to calculate our multi-layer, patch-based contrastive loss, we
        extract features from 5 layers, which are RGB pixels, the ﬁrst and
        second downsampling convolution, and the ﬁrst and the ﬁfth residual
        block. [...] For each layer’s features, we sample 256 random locations.
        """
        samples = {}

        # sample RGB pixels
        samples["rgb"] = sample_tensor(x, self.sample_size)

        out = self.conv(x)
        # print("shape after reflect and first conv", out.shape)

        out = self.dsample1(out)
        samples["dsample1"] = sample_tensor(out, self.sample_size)
        # print("shape after first downsample", out.shape)

        out = self.dsample2(out)
        samples["dsample2"] = sample_tensor(out, self.sample_size)
        # print("shape after second downsample", out.shape)

        for block_idx, res_block in enumerate(self.res_blocks):
            out = res_block(out)
            if block_idx == 0:
                samples["res_block0"] = sample_tensor(out, self.sample_size)
            elif block_idx == 4:
                samples["res_block4"] = sample_tensor(out, self.sample_size)
            # print("shape after res block", block_idx, out.shape)

        return out, samples

def sample_tensor(tensor, sample_size, replacement=False):
    """ Return a random sample of sample_size values from tensor. """

    assert type(tensor) == torch.Tensor

    flat = tensor.reshape(-1)  # flatten tensor into 1D
    size = len(flat)
    if replacement:
        # generate sample_size ints from [0, size)
        idxs = torch.randint(size, (sample_size,))
    else:
        # shuffle [0, size), then pick the first sample_size indices
        idxs = torch.randperm(size)[:sample_size]

    sample = flat[idxs]
    return sample
