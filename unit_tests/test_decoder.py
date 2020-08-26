import torch
from models.GAN.Decoder import Decoder


def test():
    in_N, in_C, in_H, in_W = 1, 256, 64, 64
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decoder_fn = Decoder().to(device)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W), device=device)
    print("Using devce", device)
    out = decoder_fn(sample_in)

    expected_shape = in_N, 3, in_H * 4, in_W * 4

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {}\n"
             + "Actual output shape: {}")
            .format(expected_shape, out.shape)
        )

    print("Decoder test passed")
    return True


if __name__ == "__main__":
    test()
