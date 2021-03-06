import torch
from models.GAN.Encoder import Encoder


def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256  # note we assume 256x256 images
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    encoder_fn = Encoder().to(device)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W), device=device)
    print("Using device", device)
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

    sample_size = 256
    for layer_name in samples:
        sample = samples[layer_name]
        print("sample shape for layer {}: {}".format(layer_name, sample.shape))

        out_N = sample.shape[0]
        out_feature_size = sample.shape[2]
        if (out_N != in_N) or (out_feature_size != sample_size):
            raise RuntimeError(
                ("Incorrect sample shape in layer {}.\n"
                 + "Expected {}, got {}")
                .format(expected_shape, sample.shape)
            )

    print("Encoder test passed")
    return True


if __name__ == "__main__":
    test()
