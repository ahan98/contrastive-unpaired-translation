import torch
from models.blocks.ResidualBlock import ResidualBlock


def test():
    in_N, in_C, in_H, in_W = 2, 3, 256, 256  # input shape
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    res_block_fn = ResidualBlock(in_C).to(device)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W), device=device)
    print("Using device", device)
    out = res_block_fn(sample_in)
    expected_shape = sample_in.shape

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {in_shape}\n"
             + "Actual output shape: {out_shape}")
            .format(in_shape=expected_shape, out_shape=out.shape)
        )

    print("Residual block test passed")
    return True


if __name__ == "__main__":
    test()
