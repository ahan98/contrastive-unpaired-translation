import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.blocks.ResidualBlock import ResidualBlock

def test():
    in_N, in_C, in_H, in_W = 2, 3, 256, 256  # input shape
    res_block_fn = ResidualBlock(in_C)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    res_block = res_block_fn(sample_in)
    out_N, out_C, out_H, out_W = res_block.shape

    if out_N != in_N:
        raise Exception(
            ("Output batch size must match input batch size.\n"
             + "Input shape: {bs_in}\n"
             + "Output shape: {bs_out}")
            .format(bs_in=in_N, bs_out=out_N)
        )
    if out_H != in_H:
        raise Exception(
            ("Output image height must match input image height.\n"
             + "Input image height: {h_in}\n"
             + "Output image height: {h_out}")
            .format(h_in=in_H, h_out=out_H)
        )
    if out_W != in_W:
        raise Exception(
            ("Output image width must match input image width.\n"
             + "Input image width: {w_in}\n"
             + "Output image width: {w_out}")
            .format(w_in=in_W, w_out=out_W)
        )

    print("Residual block test passed")
    return True

if __name__ == "__main__":
    test()
