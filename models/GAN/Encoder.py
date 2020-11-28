from abc import ABC
import torch
import torch.nn as nn
import bbml.nn


class Encoder(nn.Module, ABC):

    def __init__(self, n_res_blocks=9, batch_momentum=0.1,
                 activation_type=bbml.nn.ActivationType.RELU,
                 norm_type=bbml.nn.NormType.INSTANCE, padding_mode=bbml.nn.PaddingMode.REFLECT,
                 sample_size=256):
        super().__init__()

        self.sample_size = sample_size

        model = [
            bbml.nn.Pad2DLayer(padding=3, padding_mode=padding_mode),
            bbml.nn.Conv2DBlock(in_channels=3, out_channels=64, kernel_size=7,
                                stride=1, padding=0, batch_momentum=batch_momentum,
                                norm_type=norm_type, activation_type=activation_type),

            # Downsampling blocks
            bbml.nn.Conv2DBlock(in_channels=64, out_channels=128, kernel_size=3,
                                stride=2, padding=0, batch_momentum=batch_momentum,
                                norm_type=norm_type, activation_type=activation_type),
            bbml.nn.Conv2DBlock(in_channels=128, out_channels=256, kernel_size=3,
                                stride=2, padding=0, batch_momentum=batch_momentum,
                                norm_type=norm_type, activation_type=activation_type),

            # Residual Blocks
            bbml.nn.ResidualBlock(in_channels=26,
                                  padding_mode=padding_mode,
                                  batch_momentum=batch_momentum)
        ]

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
