import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.nn as nn
from blocks.Conv2DBlock import Conv2DBlock
from blocks.DownsamplingBlock import DownsamplingBlock
from blocks.ResidualBlock import ResidualBlock
from blocks.types import PaddingMode


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

        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks += [ResidualBlock(padding_mode=padding_mode,
                                         batch_momentum=batch_momentum)]
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        """
        From Section C.1 of Park et al.:

        In order to calculate our multi-layer, patch-based contrastive loss, we
        extract features from 5 layers, which are RGB pixels, the ﬁrst and
        second downsampling convolution, and the ﬁrst and the ﬁfth residual
        block. [...] For each layer’s features, we sample 256 random locations.
        """
        print("ENCODER FORWARD")

        samples = {}

        # sample RGB pixels
        samples["rgb"] = Encoder.__make_samples_for_tensor(x, self.sample_size)

        out = self.conv(x)
        # print("shape after reflect and first conv", out.shape)

        out = self.dsample1(out)
        samples["dsample1"] = Encoder.__make_samples_for_tensor(out, self.sample_size)
        # print("shape after first downsample", out.shape)

        out = self.dsample2(out)
        samples["dsample2"] = Encoder.__make_samples_for_tensor(out, self.sample_size)
        # print("shape after second downsample", out.shape)

        for block_idx, res_block in enumerate(self.res_blocks):
            out = res_block(out)
            print("OUT DEVICE BEFORE RES BLOCK", block_idx, out.device)
            if block_idx == 0:
                samples["res_block0"] = Encoder.__make_samples_for_tensor(out, self.sample_size)
            elif block_idx == 4:
                samples["res_block4"] = Encoder.__make_samples_for_tensor(out, self.sample_size)
            # print("shape after res block", block_idx, out.shape)

        return out, samples

    ''' Private '''

    @staticmethod
    def __make_samples_for_tensor(tensor, sample_size):
        """ Return a random sample of sample_size values from tensor. """

        assert type(tensor) == torch.Tensor and len(tensor.shape) == 4

        # Reshape from (N,C,H,W) to (N, C, H*W)
        tensor_reshape = tensor.flatten(2, 3)

        _, _, H, W = tensor.shape  # we don't explicitly need N and C
        spatial_idxs = torch.randperm(H * W)[:sample_size]

        # Extract all S sampled spatial locations across all channels and batch
        # items.
        samples = tensor_reshape[:, :, spatial_idxs]
        print("SAMPLES DEVICE", samples.device)
        return samples
