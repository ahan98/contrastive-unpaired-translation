import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.GAN.Encoder import Encoder

def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256  # note we assume 256x256 images
    encoder_fn = Encoder()
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    out, samples = encoder_fn(sample_in)

    last_residual_filter_size = 256
    expected_shape = in_N, last_residual_filter_size, in_H // 4, in_W // 4
    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {}\n"
             + "Actual output shape: {}")
            .format(expected_shape, out.shape)
        )

    expected_layers_sampled = 5
    if len(samples) != expected_layers_sampled:
        raise RuntimeError(
            ("Incorrect number of layers sampled. Expected {}, got {}")
            .format(expected_layers_sampled, len(samples))
        )

    expected_n_samples = 256
    for layer_name in samples:
        if len(samples[layer_name]) != expected_n_samples:
            raise RuntimeError(
                ("Incorrect number of samples in layer {}. Expected {}, got {}")
                .format(layer_name, expected_n_samples, len(samples))
            )

    print("Encoder test passed")
    return True

if __name__ == "__main__":
    test()