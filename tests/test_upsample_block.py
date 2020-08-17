import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
from models.upsample_block import UpsampleBlock

def test(out_channels=64):
    in_N, in_C, in_H, in_W = 2, 3, 256, 256
    upsample_block_fn = UpsampleBlock(in_C, out_channels)
    sample_in = torch.zeros((in_N, in_C, in_H, in_W))
    block = upsample_block_fn(sample_in)
    out_N, out_C, out_H, out_W = block.shape

    if out_N != in_N:
        raise Exception(
            ("Output batch size must match input batch size.\n"
             + "Input shape: {bs_in}\n"
             + "Output shape: {bs_out}")
            .format(bs_in=in_N, bs_out=out_N)
        )
    if out_C != out_channels:
        raise Exception(
            ("Output channel size must match input channel size.\n"
             + "Input shape: {in_}\n"
             + "Output shape: {output}")
            .format(in_=in_C, out=out_C)
        )
    if out_H != 2 * in_H:
        raise Exception(
            ("Output image height must be twice input image height.\n"
             + "Input image height: {h_in}\n"
             + "Output image height: {h_out}")
            .format(h_in=in_H, h_out=out_H)
        )
    if out_W != 2 * in_W:
        raise Exception(
            ("Output image width must be twice input image width.\n"
             + "Input image width: {w_in}\n"
             + "Output image width: {w_out}")
            .format(w_in=in_W, w_out=out_W)
        )
    if out_H != 2 * in_H:
        raise Exception(
            ("Output image height must be twice input image height.\n"
             + "Input image height: {h_in}\n"
             + "Output image height: {h_out}")
            .format(h_in=in_H, h_out=out_H)
        )

    print("Upsample block test passed")
    return True

if __name__ == "__main__":
    test()
