import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.blocks.UpsamplingBlock import UpsamplingBlock

def test():
    in_N, in_C, in_H, in_W = 2, 3, 256, 256
    out_C = 128
    upsample_block_fn = UpsamplingBlock(in_C, out_C)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    out = upsample_block_fn(sample_in)
    expected_shape = (in_N, out_C, 2 * in_H, 2 * in_W)

    if out.shape != expected_shape:
        raise RuntimeError(
            ("Incorrect output shape.\n"
             + "Expected output shape: {in_shape}\n"
             + "Actual output shape: {out_shape}")
            .format(in_shape=expected_shape, out_shape=out.shape)
        )

    print("Upsample block test passed")
    return True

if __name__ == "__main__":
    test()
