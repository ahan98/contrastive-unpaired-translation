import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.blocks.DownsamplingBlock import DownsamplingBlock

def test():
    in_N, in_C, in_H, in_W = 2, 3, 256, 256  # note we assume even in_H and in_W
    out_C = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    downsample_block_fn = DownsamplingBlock(in_C, out_C).to(device)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W), device=device)
    print("Using device", device)
    out = downsample_block_fn(sample_in)
    expected_shape = (in_N, out_C, 0.5 * in_H, 0.5 * in_W)

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {in_shape}\n"
             + "Actual output shape: {out_shape}")
            .format(in_shape=expected_shape, out_shape=out.shape)
        )

    print("Downsample block test passed")
    return True

if __name__ == "__main__":
    test()
