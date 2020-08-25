import torch
from models.GAN.Discriminator import Discriminator


def test():
    in_N, in_C, in_H, in_W = 1, 3, 256, 256
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    discriminator_fn = Discriminator().to(device)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W), device=device)
    print("Using device", device)
    out = discriminator_fn(sample_in)
    print(out.shape)

    expected_shape = in_N, 1, 30, 30
    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {}\n"
             + "Actual output shape: {}")
            .format(expected_shape, out.shape)
        )

    print("Discriminator test passed")
    return True


if __name__ == "__main__":
    test()
