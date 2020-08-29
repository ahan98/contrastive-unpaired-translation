import torch
import torch.nn as nn
from ..blocks.Conv2DBlock import Conv2DBlock
from ..blocks.PadLayer import PadLayer
from ..blocks.DownsamplingBlock import DownsamplingBlock
from ..blocks.ResidualBlock import ResidualBlock
from ..blocks.NormLayer import NormLayer
from ..blocks.ActivationLayer import ActivationLayer
from ..blocks.types import PaddingMode, NormType, ActivationType


class Encoder(nn.Module):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1,
                 activation_type=ActivationType.RELU,
                 norm_type=NormType.INSTANCE, padding_mode=PaddingMode.REFLECT,
                 sample_size=256):
        super().__init__()

        self.sample_size = sample_size

        use_bias = (norm_type == NormType.INSTANCE)

        ### CONV BLOCK ###

        model = [
            PadLayer(padding=3, padding_mode=padding_mode),
            nn.Conv2d(3, 64, kernel_size=7, stride=1,
                      padding=0, bias=use_bias)
        ]

        # Norm layer
        if norm_type != NormType.NONE:
            model += [NormLayer(norm_type, 64, batch_momentum)]

        # Activation layer
        if activation_type != ActivationType.NONE:
            model += [ActivationLayer(activation_type)]

        ### DOWNSAMPLING BLOCKS ###

        # First downsampling block
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=use_bias)]
        if norm_type != NormType.NONE:
            model += [NormLayer(norm_type, 128, batch_momentum)]
        if activation_type != ActivationType.NONE:
            model += [ActivationLayer(activation_type)]

        # Second downsampling block
        model += [nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=use_bias)]
        if norm_type != NormType.NONE:
            model += [NormLayer(norm_type, 256, batch_momentum)]
        if activation_type != ActivationType.NONE:
            model += [ActivationLayer(activation_type)]

        ### RESIDUAL BLOCKS ###

        for _ in range(n_res_blocks):
            model += [ResidualBlock(padding_mode=padding_mode,
                                    batch_momentum=batch_momentum)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        From Section C.1 of Park et al.:

        In order to calculate our multi-layer, patch-based contrastive loss, we
        extract features from 5 layers, which are RGB pixels, the ﬁrst and
        second downsampling convolution, and the ﬁrst and the ﬁfth residual
        block. [...] For each layer’s features, we sample 256 random locations.
        """

        samples = []
        out = x

        for layer_idx, layer_fn in enumerate(self.model):
            out = layer_fn(out)
            if layer_idx == 0:
                sample = Encoder.__make_samples_for_tensor(
                    out, self.sample_size)
                samples.append(sample)
            elif layer_idx == 4:
                sample = Encoder.__make_samples_for_tensor(
                    out, self.sample_size)
                samples.append(sample)
            elif layer_idx == 7:
                sample = Encoder.__make_samples_for_tensor(
                    out, self.sample_size)
                samples.append(sample)
            elif layer_idx == 10:
                sample = Encoder.__make_samples_for_tensor(
                    out, self.sample_size)
                samples.append(sample)
            elif layer_idx == 14:
                sample = Encoder.__make_samples_for_tensor(
                    out, self.sample_size)
                samples.append(sample)

        return out, samples

    ''' Private '''

    @staticmethod
    def __make_samples_for_tensor(tensor, sample_size):
        """ Return a random sample of sample_size values from tensor. """

        assert type(tensor) == torch.Tensor and len(tensor.shape) == 4

        # Reshape from (N,C,H,W) to (N, C, H*W)
        tensor_reshape = tensor.permute(0, 2, 3, 1).flatten(1, 2)

        image_dims_flat = tensor_reshape.shape[1]
        spatial_idxs = torch.randperm(image_dims_flat)[:sample_size]

        # Extract all S sampled spatial locations across all channels and batch
        # items.
        samples = tensor_reshape[:, spatial_idxs, :].flatten(0, 1)
        return samples
